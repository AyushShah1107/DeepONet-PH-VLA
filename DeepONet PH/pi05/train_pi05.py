"""
train_pi05.py  --  two-stage trainer for the pi0.5 SOTA-comparison variants.

Mirrors DeepONet PH/v2/train.py (reuses its generic EMA / run_stage /
save_checkpoint / loaders verbatim) but swaps in the pi0.5 policy + processor.

Variants:
    --head flow      --variant baseline             pi0.5 flow baseline
    --head deeponet  --variant baseline             pi0.5 + DeepONet
    --head deeponet  --variant ph  --lambda_ph 0.02 pi0.5 + DeepONet + PH

Frozen-backbone is the default for this campaign (stage2_steps=0 => head-only).
40-task pretrain: pass several suites via --datasets a,b,c,d (MultiLeRobotDataset).
Per-suite finetune: --dataset lerobot/libero_<suite>_image  --init <pretrain_ckpt>.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import torch

# pi0.5 policy + processor
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.datasets.factory import resolve_delta_timestamps

# reuse the proven generic training machinery from the SmolVLA v2 trainer
_V2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v2"))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from train import (  # noqa: E402
    EMA, run_stage, CSVLogger, cyclic_loader, set_seed,
)
from modeling_smolvla_ph import adapt_policy_features_to_dataset  # noqa: E402  (generic)
from modeling_pi05_deeponet import PI05DeepONetPolicy, PI05FlowPolicy  # noqa: E402

DEV = "cuda"


class _AugPreprocessor:
    """Wrap the pi0.5 preprocessor with per-sample photometric color-jitter on the
    raw RGB image tensors (observation.images.*, CHW float in [0,1]) BEFORE the base
    preprocessor normalizes them. TRAINING-ONLY: `save_pretrained` delegates to the
    base processor, so the checkpoint (and therefore eval) uses the un-augmented
    processor. Applied identically to every variant, so the flow-vs-DeepONet
    comparison stays fair — this only hardens the shared appearance robustness."""

    def __init__(self, base, brightness, contrast, saturation, hue):
        self._base = base
        from torchvision.transforms import v2 as _T
        self._cj = _T.ColorJitter(brightness=brightness, contrast=contrast,
                                  saturation=saturation, hue=hue)

    def _jitter(self, v):
        # v: (..., C, H, W) float in [0,1]; fresh jitter params per frame
        c, h, w = v.shape[-3], v.shape[-2], v.shape[-1]
        flat = v.reshape(-1, c, h, w)
        out = torch.stack([self._cj(flat[i]) for i in range(flat.shape[0])], 0)
        return out.reshape(v.shape).clamp_(0.0, 1.0)

    def __call__(self, batch):
        for k, v in list(batch.items()):
            if "observation.images" in k and torch.is_tensor(v) and v.ndim >= 3 and v.shape[-3] == 3:
                batch[k] = self._jitter(v)
        return self._base(batch)

    def __getattr__(self, name):        # delegate save_pretrained() / attrs to base
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)


def build_policy(head, variant, base_ckpt, lambda_ph, ph_k, **dk):
    ph_enabled = (variant == "ph")
    if head == "flow":
        pol = PI05FlowPolicy.from_pretrained(base_ckpt)
    elif head == "deeponet":
        pol = PI05DeepONetPolicy.from_pretrained(
            base_ckpt, ph_enabled=ph_enabled, lambda_ph=lambda_ph, ph_k=ph_k,
            deeponet_p=dk.get("deeponet_p", 256), deeponet_blocks=dk.get("deeponet_blocks", 3),
            deeponet_queries=dk.get("deeponet_queries", 8),
            deeponet_fourier=dk.get("deeponet_fourier", 16))
    else:
        raise ValueError(head)
    return pol.to(DEV)


def load_dataset(datasets, single, policy):
    names = [s.strip() for s in datasets.split(",")] if datasets else [single]
    meta = LeRobotDatasetMetadata(names[0])
    # Re-point the pretrained pi0.5 features at this dataset's cameras/state/action
    # (image/wrist_image -> iterated; state/action padded to max dims). Generic, in-place.
    adapt_policy_features_to_dataset(policy, meta)
    dt = resolve_delta_timestamps(policy.config, meta)
    if len(names) == 1:
        ds = LeRobotDataset(names[0], delta_timestamps=dt)
        stats = meta.stats
    else:                                   # 40-task pretrain across suites
        ds = MultiLeRobotDataset(names, delta_timestamps=dt)
        stats = getattr(ds, "stats", meta.stats)   # combined stats if exposed
    return ds, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["flow", "deeponet"], required=True)
    ap.add_argument("--variant", choices=["baseline", "ph"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_ckpt", default=os.environ.get("PI05_BASE", "lerobot/pi05_base"))
    ap.add_argument("--init", default=None, help="checkpoint to resume head from (finetune)")
    ap.add_argument("--dataset", default="lerobot/libero_spatial_image")
    ap.add_argument("--datasets", default=None, help="comma list for 40-task pretrain")
    ap.add_argument("--lambda_ph", type=float, default=0.02)
    ap.add_argument("--ph_k", type=int, default=8)
    ap.add_argument("--deeponet_p", type=int, default=256)
    ap.add_argument("--deeponet_blocks", type=int, default=3)
    ap.add_argument("--deeponet_queries", type=int, default=8)
    ap.add_argument("--deeponet_fourier", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    # frozen backbone => only stage1 (head-only). stage2_steps>0 unfreezes.
    ap.add_argument("--stage1_steps", type=int, default=15000)
    ap.add_argument("--stage2_steps", type=int, default=0)
    ap.add_argument("--stage1_batch", type=int, default=16)
    ap.add_argument("--stage2_batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--epoch_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    # photometric augmentation (color-jitter) — training-only, all variants identical
    ap.add_argument("--color_jitter", action="store_true",
                    help="enable per-sample color-jitter augmentation during training")
    ap.add_argument("--cj_brightness", type=float, default=0.3)
    ap.add_argument("--cj_contrast", type=float, default=0.4)
    ap.add_argument("--cj_saturation", type=float, default=0.5)
    ap.add_argument("--cj_hue", type=float, default=0.08)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"[pi05-train] head={args.head} variant={args.variant} seed={args.seed} "
          f"frozen_backbone={args.stage2_steps==0} out={out_dir}", flush=True)

    policy = build_policy(args.head, args.variant, args.base_ckpt,
                          args.lambda_ph, args.ph_k,
                          deeponet_p=args.deeponet_p, deeponet_blocks=args.deeponet_blocks,
                          deeponet_queries=args.deeponet_queries,
                          deeponet_fourier=args.deeponet_fourier)
    if args.init:
        # Resume the per-suite finetune from the 40-task pretrain for BOTH heads.
        # (Was gated to deeponet only -> the flow baseline silently discarded its
        #  pretrain and warm-started from stock pi05_base, an unfair asymmetric init.)
        src_cls = PI05DeepONetPolicy if args.head == "deeponet" else PI05FlowPolicy
        sd = src_cls.from_pretrained(args.init).state_dict()
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        print(f"[pi05-train] resumed {args.head} head from {args.init} "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    dataset, norm_stats = load_dataset(args.datasets, args.dataset, policy)
    # pi0.5 defaults STATE/ACTION to QUANTILES (needs q01/q99 the LeRobot LIBERO
    # datasets don't carry). With a frozen backbone + freshly-trained head, MEAN_STD
    # is valid and consistent across train/eval. Override so the existing stats work.
    from lerobot.configs.types import NormalizationMode
    for k in ("STATE", "ACTION"):
        if policy.config.normalization_mapping.get(k) == NormalizationMode.QUANTILES:
            policy.config.normalization_mapping[k] = NormalizationMode.MEAN_STD
    preprocessor, _post = make_pi05_pre_post_processors(policy.config, dataset_stats=norm_stats)
    if args.color_jitter:
        preprocessor = _AugPreprocessor(preprocessor, args.cj_brightness, args.cj_contrast,
                                        args.cj_saturation, args.cj_hue)
        print(f"[pi05-train] color-jitter aug ON (b={args.cj_brightness} c={args.cj_contrast} "
              f"s={args.cj_saturation} h={args.cj_hue}) — TRAIN only, eval uses base processor", flush=True)
    n_frames = getattr(dataset, "num_frames", len(dataset))
    print(f"[pi05-train] dataset frames={n_frames}", flush=True)

    ema = EMA(policy, decay=args.ema) if args.ema and args.ema > 0 else None
    cfg = {
        "stage1": dict(steps=args.stage1_steps, batch=args.stage1_batch, head_lr=args.head_lr),
        "stage2": dict(steps=args.stage2_steps, batch=args.stage2_batch, head_lr=args.head_lr,
                       backbone_lr=args.backbone_lr, warmup=args.warmup, grad_ckpt=True),
    }
    step_fields = ["global_step", "stage", "dataset_epoch", "flow_matching_loss", "l1_loss",
                   "ph_loss", "total_loss", "lr_head", "lr_backbone", "vram_gb",
                   "grad_norm", "step_time_s"]
    epoch_fields = ["epoch_global_step", "stage", "logging_epoch", "flow_matching_loss",
                    "l1_loss", "ph_loss", "total_loss", "vram_gb"]
    step_logger = CSVLogger(out_dir / "log_step.csv", step_fields)
    epoch_logger = CSVLogger(out_dir / "log_epoch.csv", epoch_fields)
    (out_dir / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "stage_configs": cfg, "dataset_frames": n_frames}, indent=2, default=str))

    t0 = time.perf_counter(); gstep = 0
    di = cyclic_loader(dataset, cfg["stage1"]["batch"], args.num_workers)
    gstep = run_stage("stage1", cfg["stage1"], policy, ema, di, preprocessor, out_dir,
                      step_logger, epoch_logger, args.epoch_steps, args.ckpt_every, gstep)
    if cfg["stage2"]["steps"] > 0:
        di = cyclic_loader(dataset, cfg["stage2"]["batch"], args.num_workers)
        gstep = run_stage("stage2", cfg["stage2"], policy, ema, di, preprocessor, out_dir,
                          step_logger, epoch_logger, args.epoch_steps, args.ckpt_every, gstep)
    print(f"\n[pi05-train] DONE steps={gstep} wall={(time.perf_counter()-t0)/60:.1f} min "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

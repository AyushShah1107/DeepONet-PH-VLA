#!/usr/bin/env python
"""
evaluate.py  (DeepONet campaign)
================================
Closed-loop success-rate evaluation for the head-to-head models on LIBERO-Spatial
(in-distribution) and LIBERO-V (viewpoint / lighting / sensor-noise perturbations).

Receding-horizon control: each model predicts a 50-step chunk, executes the first
`--replan` actions open-loop (default 5), then re-observes and re-predicts. This
is enforced uniformly for all models by setting config.n_action_steps = replan,
so flow-matching (M1) and DeepONet (M3/M4) are compared under identical control.

Models are specified as repeatable  --model NAME=HEAD=CKPT  (HEAD in {flow,deeponet}),
e.g.  --model m1=flow=runs/m1/checkpoints/LATEST
      --model m3=deeponet=runs/m3/checkpoints/LATEST
      --model m4=deeponet=runs/m4/checkpoints/LATEST

Results stream to <out>/success_rates.json (resumable).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import re
import numpy as np
import torch
import imageio.v2 as imageio

os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from libero_v_wrapper import make_libero_v_env, _make_base_libero_env
from act_common import load_ckpt

DEV = os.environ.get("EVAL_DEVICE", "cpu")   # CPU-safe copy: never opens a CUDA context
REPO_DATA = "lerobot/libero_spatial_image"
PERTURBATIONS = ("viewpoint", "lighting", "sensor_noise")
N_TASKS = 10
DEFAULT_MAX_STEPS = 520


def resolve_latest(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if p.name in ("LATEST", "BEST"):
        ptr = p.parent / f"{p.name}.txt"
        if not ptr.exists() and p.name == "BEST":
            ptr = p.parent / "LATEST.txt"
        if ptr.exists():
            return str(p.parent / ptr.read_text().strip())
    return str(p)


def load_policy(ckpt: str, meta, replan: int):
    ckpt = resolve_latest(ckpt)
    policy, variant = load_ckpt(meta, ckpt)
    policy = policy.to(DEV).eval()
    policy.config.n_action_steps = replan  # receding-horizon replan interval
    pre, post = make_act_pre_post_processors(policy.config, dataset_stats=meta.stats)
    return policy, pre, post


# --------------------------------------------------------------------------- obs conv
def _to_t(x):
    return torch.as_tensor(np.asarray(x)).float()


def _quat2axisangle(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(torch.float32)
    B = quat.shape[0]
    w = quat[:, 3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    out = torch.zeros((B, 3))
    mask = den > 1e-10
    if mask.any():
        angle = 2.0 * torch.acos(w[mask])
        axis = quat[mask, :3] / den[mask].unsqueeze(1)
        out[mask] = axis * angle.unsqueeze(1)
    return out


def env_obs_to_policy_input(obs, task_description):
    pix = obs["pixels"]
    rs = obs["robot_state"]

    def img(a):
        t = torch.as_tensor(np.asarray(a)).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return torch.flip(t, dims=[2, 3])

    eef_pos = _to_t(rs["eef"]["pos"]).reshape(1, 3)
    eef_quat = _to_t(rs["eef"]["quat"]).reshape(1, 4)
    grip = _to_t(rs["gripper"]["qpos"]).reshape(1, 2)
    state = torch.cat([eef_pos, _quat2axisangle(eef_quat), grip], dim=-1)
    return {
        "observation.images.image": img(pix["image"]),
        "observation.images.wrist_image": img(pix["image2"]),
        "observation.state": state,
        "task": [task_description],
    }


@torch.no_grad()
def rollout(policy, pre, post, env, task_description, max_steps, seed, capture=False):
    policy.reset()
    obs, info = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        if capture:
            frames.append(np.asarray(obs["pixels"]["image"])[::-1].copy())
        pin = pre(env_obs_to_policy_input(obs, task_description))
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            action = policy.select_action(pin)
        action = post(action)
        a = action.to("cpu").float().numpy().reshape(-1)
        obs, reward, terminated, truncated, info = env.step(a)
        if info.get("is_success", False):
            if capture:
                frames.append(np.asarray(obs["pixels"]["image"])[::-1].copy())
            return True, frames
        if terminated or truncated:
            break
    return False, frames


def _sanitize(name, maxlen=45):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:maxlen]


def _save(results, out_path):
    Path(out_path).write_text(json.dumps(results, indent=2))


def eval_indist(models, n_episodes, max_steps, results, out_path, suite, bench_name,
                n_tasks=N_TASKS, test_seeds=3, task_start=0):
    """In-distribution eval over `test_seeds` independent seeds -> per-task mean +/- std."""
    bench = results.setdefault(bench_name, {})
    vid_root = Path(out_path).parent / "episode_videos"
    for mname, (policy, pre, post) in models.items():
        per_model = bench.setdefault(mname, {"per_task": {}, "average": None, "average_std": None})
        for task_id in range(task_start, task_start + n_tasks):
            key = str(task_id)
            if key in per_model["per_task"]:
                continue
            env = _make_base_libero_env(task_id, suite_name=suite)
            task_desc = env.task_description
            vdir = vid_root / mname; vdir.mkdir(parents=True, exist_ok=True)
            seed_rates = []
            for sd in range(test_seeds):
                succ = []
                for ep in range(n_episodes):
                    cap = False   # CPU run: no video capture (success unaffected; saves CPU+disk)
                    ok, frames = rollout(policy, pre, post, env, task_desc, max_steps,
                                         seed=1000 + 1000 * sd + ep, capture=cap)
                    succ.append(ok)
                    if cap:
                        fn = f"task{task_id}_{_sanitize(task_desc)}_ep{ep:02d}_{'PASS' if ok else 'FAIL'}.mp4"
                        try:
                            imageio.mimwrite(str(vdir / fn), frames, fps=20, codec="libx264",
                                             macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
                        except Exception as e:
                            print(f"[vid] save failed {fn}: {e}", flush=True)
                seed_rates.append(float(np.mean(succ)))
            env.close()
            mean, std = float(np.mean(seed_rates)), float(np.std(seed_rates))
            per_model["per_task"][key] = {"task": task_desc, "success_rate": mean, "std": std,
                                          "per_seed": seed_rates, "n": n_episodes, "seeds": test_seeds}
            print(f"[{bench_name}] {mname} task{task_id}: {mean*100:.1f}% +/- {std*100:.1f}  "
                  f"(seeds {[round(r*100) for r in seed_rates]})", flush=True)
            _save(results, out_path)
        rates = [v["success_rate"] for v in per_model["per_task"].values()]
        stds = [v["std"] for v in per_model["per_task"].values()]
        per_model["average"] = float(np.mean(rates)) if rates else None
        per_model["average_std"] = float(np.mean(stds)) if stds else None
        _save(results, out_path)


def eval_pert(models, n_episodes, max_steps, results, out_path, suite, bench_name, n_tasks=N_TASKS):
    bench = results.setdefault(bench_name, {})
    for mname, (policy, pre, post) in models.items():
        per_model = bench.setdefault(mname, {})
        for pert in PERTURBATIONS:
            ptab = per_model.setdefault(pert, {"per_task": {}, "average": None})
            for task_id in range(n_tasks):
                key = str(task_id)
                if key in ptab["per_task"]:
                    continue
                env = make_libero_v_env(task_id, perturbation=pert, seed=task_id, suite_name=suite)
                task_desc = env.env.task_description
                succ = [rollout(policy, pre, post, env, task_desc, max_steps, seed=2000 + ep)[0]
                        for ep in range(n_episodes)]
                env.close()
                rate = float(np.mean(succ))
                ptab["per_task"][key] = {"task": task_desc, "success_rate": rate, "n": n_episodes}
                print(f"[{bench_name}/{pert}] {mname} task{task_id}: {rate*100:.1f}%", flush=True)
                _save(results, out_path)
            rates = [v["success_rate"] for v in ptab["per_task"].values()]
            ptab["average"] = float(np.mean(rates)) if rates else None
            _save(results, out_path)
        present = [k for k in PERTURBATIONS
                   if isinstance(per_model.get(k), dict) and per_model[k].get("average") is not None]
        pavgs = [per_model[k]["average"] for k in present]
        per_model["average_over_perturbations"] = float(np.mean(pavgs)) if pavgs else None
        _save(results, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="repeatable NAME=CKPT (variant read from the checkpoint meta.json)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--dataset", default=REPO_DATA)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--replan", type=int, default=5, help="receding-horizon replan interval")
    ap.add_argument("--indist_episodes", type=int, default=10)
    ap.add_argument("--test_seeds", type=int, default=3)
    ap.add_argument("--pert_episodes", type=int, default=8)
    ap.add_argument("--n_tasks", type=int, default=N_TASKS)
    ap.add_argument("--task_start", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--only", choices=["indist", "pert", "both"], default="indist")
    ap.add_argument("--perturbations", default=None)
    args = ap.parse_args()

    global PERTURBATIONS
    if args.perturbations:
        PERTURBATIONS = tuple(p.strip() for p in args.perturbations.split(",") if p.strip())

    indist_name = args.suite.replace("libero_", "LIBERO-").upper()
    pert_name = f"{indist_name}-V"

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / "success_rates.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    meta = LeRobotDatasetMetadata(args.dataset)
    norm_stats = torch.load(args.stats_path) if args.stats_path else meta.stats

    models = {}
    for spec in args.model:
        name, ckpt = spec.split("=", 1)
        models[name] = load_policy(ckpt, meta, args.replan)
        print(f"[eval] loaded {name} replan={args.replan} seeds={args.test_seeds} <- {resolve_latest(ckpt)}", flush=True)

    results.setdefault("_config", {})[args.suite] = {
        "replan": args.replan, "indist_episodes": args.indist_episodes, "test_seeds": args.test_seeds,
        "max_steps": args.max_steps,
        "models": {n: resolve_latest(s.split("=", 1)[1]) for n, s in zip(models, args.model)},
    }

    if args.only in ("indist", "both"):
        eval_indist(models, args.indist_episodes, args.max_steps, results, out_path,
                    args.suite, indist_name, n_tasks=args.n_tasks, test_seeds=args.test_seeds, task_start=args.task_start)
    if args.only in ("pert", "both"):
        eval_pert(models, args.pert_episodes, args.max_steps, results, out_path,
                  args.suite, pert_name, n_tasks=args.n_tasks)

    summary = {}
    for m in models:
        summary[m] = {
            f"{indist_name}_avg": results.get(indist_name, {}).get(m, {}).get("average"),
            f"{pert_name}_avg": results.get(pert_name, {}).get(m, {}).get("average_over_perturbations"),
        }
    results["_summary"] = summary
    _save(results, out_path)
    print("[eval] DONE. summary:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

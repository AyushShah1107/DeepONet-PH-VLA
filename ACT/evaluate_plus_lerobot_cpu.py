#!/usr/bin/env python
"""
evaluate_plus_lerobot.py
========================
CORRECTED LIBERO-Plus eval that drives the policy through LeRobot's official
`LiberoEnv` (create_libero_envs, control_mode='relative') — the SAME env/action
convention the policy was trained under — instead of a hand-rolled raw-robosuite
OffScreenRenderEnv. This matches the colleague's AEGIS harness (plus_eval_aegis.py)
that reproduces ACT base ~51% on Object-Plus, where our raw-env harness gave ~31%.

Works for every variant (act / act_deeponet / act_deeponet_ph) via act_common.load_ckpt.
Instruction is cleaned from the classification task name for all categories EXCEPT
Language Instructions (where the reworded prompt IS the perturbation).
"""
from __future__ import annotations
import os, sys, json, argparse, re
from collections import defaultdict
from pathlib import Path

# --- LIBERO-Plus must be on sys.path BEFORE lerobot imports `libero` ---
REPO = os.environ.get("LIBERO_PLUS_REPO", "/home/user/Desktop/Ayush PH test/third_party/LIBERO-plus")
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_plus"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import libero.libero.envs.env_wrapper  # noqa: F401  (before torch — linker gotcha)
import numpy as np
np.float_ = getattr(np, "float_", np.float64)
np.complex_ = getattr(np, "complex_", np.complex128)
import torch as _torch
_orig_load = _torch.load
def _compat_load(*a, **k):
    k.setdefault("weights_only", False); return _orig_load(*a, **k)
_torch.load = _compat_load

import gymnasium as gym
import torch

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.libero import create_libero_envs
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from act_common import load_ckpt

DEV = os.environ.get("EVAL_DEVICE", "cpu")   # CPU-safe copy: never opens a CUDA context
CATEGORIES = ["Camera Viewpoints", "Light Conditions", "Sensor Noise", "Background Textures",
              "Objects Layout", "Robot Initial States", "Language Instructions"]
_PERT = re.compile(r"_(table|view|add|light|noise|language|initstate)_\d+.*$")
_SCENE = re.compile(r"^[A-Z][A-Za-z_]*?SCENE\d+_")


def clean_instruction(name):
    return _SCENE.sub("", _PERT.sub("", name)).replace("_", " ").strip()


def _to_t(a):
    return torch.as_tensor(np.asarray(a)).float()


def _quat2axisangle(quat):
    quat = quat.to(torch.float32)
    w = quat[:, 3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    out = torch.zeros((quat.shape[0], 3))
    m = den > 1e-10
    if m.any():
        out[m] = (quat[m, :3] / den[m].unsqueeze(1)) * (2.0 * torch.acos(w[m])).unsqueeze(1)
    return out


def env_obs_to_policy_input(obs, instr):
    pix, rs = obs["pixels"], obs["robot_state"]

    def img(a):
        t = _to_t(a) / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return torch.flip(t, dims=[2, 3])

    state = torch.cat([_to_t(rs["eef"]["pos"]).reshape(1, 3),
                       _quat2axisangle(_to_t(rs["eef"]["quat"]).reshape(1, 4)),
                       _to_t(rs["gripper"]["qpos"]).reshape(1, 2)], dim=-1)
    return {"observation.images.image": img(pix["image"]),
            "observation.images.wrist_image": img(pix["image2"]),
            "observation.state": state, "task": [instr]}


def _devec(obs):
    return {"pixels": {"image": obs["pixels"]["image"][0], "image2": obs["pixels"]["image2"][0]},
            "robot_state": {"eef": {"pos": obs["robot_state"]["eef"]["pos"][0],
                                    "quat": obs["robot_state"]["eef"]["quat"][0]},
                            "gripper": {"qpos": obs["robot_state"]["gripper"]["qpos"][0]}}}


@torch.no_grad()
def rollout(policy, pre, post, env, instr, max_steps):
    policy.reset()
    obs, info = env.reset()
    for _ in range(max_steps):
        pin = pre(env_obs_to_policy_input(_devec(obs), instr))
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            action = policy.select_action(pin)
        a = post(action).to("cpu").float().numpy().reshape(1, -1)
        obs, r, term, trunc, info = env.step(a)
        fi = info.get("final_info")
        isk = info.get("is_success")
        if (isinstance(fi, dict) and bool(np.asarray(fi.get("is_success", False)).any())) or \
           (isk is not None and bool(np.asarray(isk).any())):
            return True
        if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
            break
    return False


def build_env(suite, tid, max_steps):
    rep = LiberoEnv(task=suite, fps=10, episode_length=max_steps)
    gk = dict(rep.gym_kwargs); gk["task_ids"] = [tid]
    return create_libero_envs(task=suite, n_envs=1, camera_name=rep.camera_name,
                              init_states=False, gym_kwargs=gk, env_cls=gym.vector.SyncVectorEnv,
                              control_mode=rep.control_mode, episode_length=max_steps)[suite][tid]


def stratified_sample(tasks, n, seed=42):
    import random
    rng = random.Random(seed)
    by_d = defaultdict(list)
    for t in tasks:
        by_d[t["difficulty_level"]].append(t)
    for v in by_d.values():
        v.sort(key=lambda t: t["id"]); rng.shuffle(v)
    out, levels, i = [], sorted(by_d, key=lambda d: (d is None, d)), 0
    while len(out) < min(n, len(tasks)):
        lv = levels[i % len(levels)]
        if by_d[lv]:
            out.append(by_d[lv].pop())
        i += 1
        if all(len(by_d[lv]) == 0 for lv in levels):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=CKPT")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--n_per_cat", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--categories", default=None)
    args = ap.parse_args()
    cats = CATEGORIES if not args.categories else [c.strip() for c in args.categories.split(",")]

    tc = json.load(open(f"{REPO}/libero/libero/benchmark/task_classification.json"))[args.suite]
    by_cat = defaultdict(list)
    for t in tc:
        by_cat[t["category"]].append({**t, "index": t["id"]})
    sampled = {c: stratified_sample(by_cat[c], args.n_per_cat, seed=args.seed) for c in cats}
    for c in cats:
        print(f"[plus] {c}: sampled {len(sampled[c])}/{len(by_cat[c])}", flush=True)

    meta = LeRobotDatasetMetadata(args.dataset)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / "robustness_plus.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("_config", {}).update({"suite": args.suite, "replan": args.replan,
        "n_per_cat": args.n_per_cat, "max_steps": args.max_steps, "seed": args.seed,
        "env": "lerobot_LiberoEnv_relative", "categories": cats})
    out_path.write_text(json.dumps(results, indent=2))

    for spec in args.model:
        name, ckpt = spec.split("=", 1)
        policy, variant = load_ckpt(meta, ckpt)
        policy = policy.to(DEV).eval()
        policy.config.n_action_steps = args.replan
        pre, post = make_act_pre_post_processors(policy.config, dataset_stats=meta.stats)
        print(f"[plus] loaded {name} (variant={variant}) <- {ckpt}", flush=True)
        mres = results.setdefault(name, {})
        for c in cats:
            cres = mres.setdefault(c, {"per_task": {}, "average": None})
            is_lang = (c == "Language Instructions")
            for t in sampled[c]:
                key = str(t["index"])
                if key in cres["per_task"]:
                    continue
                env = build_env(args.suite, t["index"], args.max_steps)
                instr = env.envs[0].task_description if is_lang else clean_instruction(t["name"])
                ok = rollout(policy, pre, post, env, instr, args.max_steps)
                env.close()
                cres["per_task"][key] = {"name": t["name"], "difficulty": t["difficulty_level"], "success": bool(ok)}
                cres["average"] = float(np.mean([v["success"] for v in cres["per_task"].values()]))
                out_path.write_text(json.dumps(results, indent=2))
                print(f"[plus/{name}/{c[:12]}] idx{t['index']}: {'OK' if ok else 'x'} (cat {cres['average']*100:.0f}%)", flush=True)
            cat_avgs = [mres[c]["average"] for c in cats if mres.get(c, {}).get("average") is not None]
            mres["robustness_average"] = float(np.mean(cat_avgs)) if cat_avgs else None
            out_path.write_text(json.dumps(results, indent=2))
        print(f"[plus] {name} robustness avg = {(mres.get('robustness_average') or 0)*100:.1f}%", flush=True)
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("[plus] DONE", flush=True)


if __name__ == "__main__":
    main()

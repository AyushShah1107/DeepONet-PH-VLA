# Comp‑3 — DeepONet operator head on GR00T N1.6‑3B (LIBERO)

Scaling the DeepONet operator action head (and the PH loss) from SmolVLA/ACT up to
**NVIDIA GR00T N1.6‑3B** (Eagle VLM + diffusion action head). Head‑only (frozen 3B
backbone), mirroring the comp‑1 (pi0.5) protocol: 40‑task combined **pretrain (15K)** →
per‑suite **finetune (15K)** → **in‑dist + LIBERO‑Plus** eval on Spatial/Object/Long/Goal.

Three variants: `c3_groot` (native diffusion head, baseline) · `c3_groot_deeponet` ·
`c3_groot_deeponet_ph`.

## Results — closed‑loop, single seed (`results_c3/`) — COMPLETE

| Suite | Diffusion in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 98.3 / **91.1** | 96.7 / 83.9 | 98.3 / 83.9 |
| Object | 98.3 / 83.9 | 99.2 / 75.0 | 99.2 / **82.1** |
| Long | 95.8 / **91.1** | 95.0 / 69.6 | 87.5 / 73.2 |
| Goal | 99.2 / 83.9 | 96.7 / 75.0 | 97.5 / 73.2 |
| **Average** | **97.9 / 87.5** | 96.9 / 75.9 | 95.6 / 78.1 |

**Honest result: the operator head essentially *ties* in‑distribution (96.9 vs 97.9) at a ~100× smaller head
(~11 M vs the 1.09 B DiT), but loses robustness (−11.6 pp: 75.9 vs 87.5).** `+PH` partly recovers it (78.1,
−9.4 vs baseline), mainly by lifting Object‑Plus 75.0 → 82.1. GR00T is only *partly* frozen (projector + DiT
train, Eagle VLM mostly frozen), so its robustness loss sits **between** adapting SmolVLA (+20.6) and fully‑frozen
pi0.5 (−13.4 aug) — the middle point of the co‑adaptation trend. The 3 B diffusion baseline is inherently
appearance‑robust (Spatial breakdown: Light/Sensor/Texture/Layout/Language 100; Camera 62.5, Robot‑init 75.0 —
appearance‑robust, geometry‑fragile), leaving little robustness headroom. Single seed; 8 ep/category × 7 = 56
LIBERO‑Plus instances/suite. Raw per‑category rates in `results_c3/_<variant>__<suite>_raw/robustness_plus.json`.

**Checkpoints:** none retained — the 3 B per‑suite finetunes were pruned after each confirmed result (below), and
no GR00T checkpoint was small enough to archive. The `results_c3/` JSONs are the durable record; re‑create via
`download_ckpt.py` + `run_c3_groot.sh`.

## Files
| File | What |
|---|---|
| `run_c3_groot.sh` | master runner (prep → self‑smoke gate → 3 variants), single‑GPU, GPU‑guarded |
| `eval_groot.py` | closed‑loop LIBERO eval **client** (project venv); drives sim, queries the GR00T PolicyServer |
| `download_ckpt.py` | fetch the GR00T‑N1.6‑3B base checkpoint |
| `gr00t_head_code/` | the new GR00T action‑head modules: `deeponet_action_head.py`, `deeponet_head_v2.py`, `ph_loss.py` |
| `gr00t_upstream_patches.diff` | the in‑place edits to upstream Isaac‑GR00T (`gr00t_n1d6.py`, `eagle_backbone.py`, `launch_finetune.py`) |
| `run_c1_replan5.sh` | comp‑1 (pi0.5) replan=5 re‑campaign runner (kept here for provenance) |

## Architecture (server/client split)
- **Server** (gr00t conda env, transformers 4.51.3): the 3B model + head, served over ZMQ via
  `run_gr00t_server.py`. The action head is **env‑driven at every load** — `GR00T_ACTION_HEAD`
  ∈ {diffusion, deeponet, deeponet_ph} must be set at pretrain, finetune, **and** the eval server,
  or the trained head is silently discarded. `GR00T_PH=1` enables the PH loss.
- **Client** (project venv, py3.12): `eval_groot.py` owns robosuite + original LIBERO (in‑dist) and
  LIBERO‑Plus (robustness), and re‑uses one obs/action code path for both.

## Environment (GR00T's pinned deps — this matters)
The Eagle backbone **hard‑requires flash_attention_2** and bf16. The pins that work:
`torch==2.7.1+cu128`, `torchvision==0.22.1`, `torchcodec==0.4.0`, `numpy==1.26.4`,
`transformers==4.51.3`, `flash-attn==2.8.0.post2` (cu12torch2.7cxx11abiTRUE‑cp310 wheel).
Run everything with `PYTHONNOUSERSITE=1` so a `~/.local` transformers can't shadow the env.

## Bring‑up: 7 fixes to pass the self‑smoke gate
The GR00T DeepONet pipeline had never been validated end‑to‑end. The self‑smoke gate
(tiny real‑batch train → server → 1‑ep eval) surfaced them one by one:

1. **transformers** 4.57.6 → **4.51.3** (4.57 crashes the Eagle config on `_attn_implementation_autoset`); `PYTHONNOUSERSITE=1` to defeat `~/.local` shadowing.
2. **Dataset**: a rate‑limited download left `libero_spatial` with 163/432 wrist videos while reporting success — re‑fetch until the wrist count equals the episode count.
3. **Head state‑shape** (our code): GR00T state is `(B, T_state, dim)`, not `(B, dim)` — append state tokens along the sequence dim instead of `unsqueeze`.
4. **Backbone NaN**: torch 2.10+cu130 / flash‑attn 2.8.3 ABI mismatch made the Eagle backbone emit NaN. Fixed by aligning to GR00T's pins (above).
5. **Uninitialized head params**: `from_pretrained` fast‑init leaves new head params (e.g. `state_proj`) as meta‑materialized garbage (~1e32). Added `_init_weights` to `Gr00tN1d6` — transformers calls it only on the missing (head) keys.
6. **Server marker undetected**: the `[GR00T] action head = deeponet` confirmation is a `print()` to block‑buffered stdout, trapped while the server blocks in `recv()`. Fixed with `python -u` / `PYTHONUNBUFFERED=1` (the correct head was loading all along).
7. **In‑dist eval init‑states**: LIBERO's `get_task_init_states()` → `torch.load` fails under torch≥2.6's `weights_only=True` on numpy globals. Added a `weights_only=False` shim (trusted local files).

Only #3 was our own code; the rest were environment drift or torch‑2.6 behavior changes.

## Status
Started 2026‑07‑05, **COMPLETE 2026‑07‑08** (all 3 variants × 4 suites). Ran autonomously;
checkpoints were pruned after each suite's eval (data‑loss‑safe: prune only after a confirmed
non‑null result JSON), so `results_c3/` holds `<variant>__<suite>.json` + `_<variant>__<suite>_raw/`
(per‑category rates) as the durable record. Raw results also mirrored to HF
`AyushShah1107/pi05-deeponet-libero` under `results_c3/`.

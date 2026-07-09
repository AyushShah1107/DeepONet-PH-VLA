# Comp‑1 — DeepONet operator head on pi0.5 (LIBERO)

Scaling the DeepONet operator action head (+ PH loss) to **pi0.5 (3.3 B, PaliGemma VLM + flow‑matching
action expert)**. Head‑only (frozen backbone): 40‑task combined **pretrain (15K)** → per‑suite **finetune
(15K)** → **in‑dist + LIBERO‑Plus** on Spatial/Object/Long/Goal. Three variants: `c1_pi05` (flow baseline) ·
`c1_pi05_deeponet` · `c1_pi05_deeponet_ph`.

Code: `train_pi05.py`, `eval_pi05.py`, `modeling_pi05_deeponet.py`. Runner: `../comp3_groot/run_c1_replan5.sh`
(aug: `run_c1_aug.sh`, `--color_jitter`).

## Results — **canonical: photometric‑augmentation run**, closed‑loop replan=5, single seed (`results_aug/`)

Every head is trained with identical per‑sample color‑jitter (`--color_jitter`, training‑only, applied to all
three heads through the same wrapper) — the standard way to buy robustness on a **frozen** backbone without
unfreezing it. This is the final pi0.5 result.

| Suite | Flow in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 87.5 / **71.4** | 92.5 / 44.6 | 90.8 / 35.7 |
| Object | 88.3 / **57.1** | 92.5 / 55.4 | 85.8 / 53.6 |
| Long | 70.0 / **46.4** | 53.3 / 26.8 | 44.2 / 16.1 |
| Goal | 90.8 / **55.4** | 97.5 / 50.0 | 93.3 / 41.1 |
| **Average** | **84.2 / 57.6** | **84.0 / 44.2** | 78.5 / 36.6 |

**Honest result (aug): the operator head *ties* flow in‑distribution (84.0 vs 84.2, Δ −0.2) but flow still
leads robustness (+13.4 pp: 57.6 vs 44.2); adding the shape regularizer (+SR) makes it *worse* (−21.0: 36.6),
not better.** Augmentation lifts both heads relative to the no‑aug run (flow Plus 54.9 → 57.6; DeepONet 29.5 →
44.2) and **halves** the robustness gap (−25.4 → −13.4 pp), but it does **not flip the sign**. The SmolVLA/ACT
robustness win still does not transfer to a frozen pi0.5 backbone. Single seed; per‑suite deltas are noisy
(Object is within noise at −1.8 pp; Spatial −26.8 and Long −19.6 carry the gap). **+SR *hurts* on the frozen
backbone** (36.6 Plus vs the bare head's 44.2, and 78.5 in‑dist vs 84.0) — the opposite of its effect on the
adapting backbones (ACT/SmolVLA), consistent with co‑adaptation: a training‑only shape prior cannot help a
representation that is not allowed to adapt.

### Why — frozen backbone (unchanged by augmentation)
On pi0.5 the backbone is **frozen and head‑only**, and was pretrained end‑to‑end *as a flow model*, so its
features are native to flow and foreign to a bolted‑on operator head. The operator head's benefit is
representation‑level, and it can't reshape a frozen representation — augmentation regularizes the readout but
does not let the head co‑adapt the backbone. The signature fits: **in‑dist ties** (the readout still fits) while
**Plus still trails** (OOD robustness lives in the representation). In the cases where the operator head *won*
robustness (ACT trains end‑to‑end; SmolVLA unfreezes stage‑2 at lr 1e‑5), the representation could co‑adapt.
**The decisive follow‑up is unfreezing the pi0.5 backbone** (matching the SmolVLA regime) on 1–2 suites.

## Provenance — earlier protocols (superseded, kept for the record)

Two earlier pi0.5 campaigns preceded the augmentation run. Both are retained under `results_replan5/` and
`results_openloop_superseded/`; the augmentation table above is canonical.

**(1) replan=5, no augmentation** (`results_replan5/`) — the corrected‑protocol run *before* augmentation:

| Suite | Flow in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 88.3 / 64.3 | 86.7 / 39.3 | 90.0 / 48.2 |
| Object | 87.5 / 55.4 | 81.7 / 30.4 | 83.3 / 32.1 |
| Long | 66.7 / 50.0 | 60.8 / 12.5 | 44.2 / 12.5 |
| Goal | 93.3 / 50.0 | 96.7 / 35.7 | 87.5 / 50.0 |
| **Average** | **84.0 / 54.9** | **81.5 / 29.5** | **76.3 / 35.7** |

Same conclusion, larger gap (flow +25.4 pp on Plus). Augmentation is what closed it to −13.4.

**(2) open‑loop (`n_action_steps=50`), superseded** (`results_openloop_superseded/`) — the original run, before
the replan bug was caught:

| Suite | Flow in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 53.3 / 33.9 | 63.3 / 19.6 | 70.0 / 16.1 |
| Object | 58.3 / 28.6 | 41.7 / 7.1 | 47.5 / 8.9 |
| Long | 28.3 / 14.3 | 30.8 / 5.4 | 30.8 / 7.1 |
| Goal | 71.7 / 33.9 | 77.5 / 32.1 | 78.3 / 26.8 |
| **Average** | **52.9 / 27.7** | **53.3 / 16.1** | **56.7 / 14.7** |

**The replan bug.** The first run used the model config's default `n_action_steps=50`: it executed all 50
predicted actions **open‑loop** before re‑querying, instead of the **replan=5** receding‑horizon control the
SmolVLA study this campaign compares against uses. Near‑open‑loop control tanks closed‑loop LIBERO, so every
absolute score was artificially low (a second audit also caught an asymmetric `--init` where the flow baseline
discarded its pretrain). `eval_pi05.py` now forces `n_action_steps = args.replan` (=5) and both heads resume the
same pretrain; the campaign was re‑run → `results_replan5/`, then re‑run with augmentation → `results_aug/`.

**What changed across all three protocols — and what didn't.** The replan fix roughly *doubled* every absolute
score (flow in‑dist 52.9 → 84.0); augmentation then lifted robustness across the board (flow Plus 54.9 → 57.6,
DeepONet 29.5 → 44.2). But the **conclusion is invariant**: on a frozen pi0.5 backbone the flow baseline leads
robustness under **all three** protocols (Plus average — flow 27.7 vs DeepONet 16.1 open‑loop; 54.9 vs 29.5 at
replan=5; **57.6 vs 44.2 with augmentation**). The negative result is a property of the frozen head‑only regime,
not an artifact of the replan bug or of missing augmentation.

## Checkpoints
The un‑augmented deeponet_ph 40‑task pretrain (16 GB) is on HF `AyushShah1107/pi05-deeponet-libero` under
`checkpoints/`. The **augmentation** deeponet_ph 40‑task pretrain (16.6 GB) is preserved locally at
`outputs_aug/c1_pi05_deeponet_ph_aug/pretrain/checkpoints/15000/`; HF mirror pending. Per‑suite finetune
checkpoints are pruned after each eval (only ever after a confirmed result JSON), so the finetuned aug weights
are not retained — the raw per‑episode results (`results_aug/`, mirrored to HF) are the durable record.

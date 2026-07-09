# DeepONet Operator Action Head + Persistent-Homology Loss for Vision-Language-Action Models

A research project that replaces the heavy **flow-matching action expert** of a Vision-Language-Action (VLA) model
(SmolVLA) with a lightweight **DeepONet operator action head**, optionally regularised by a
**Persistent-Homology (PH) topological loss**, and benchmarks it head-to-head on **LIBERO** for
in-distribution accuracy, **robustness** (LIBERO-Plus, 7 perturbations), parameter count, and inference latency.

> **One-line result:** the DeepONet head is **~9.6× smaller** (99.9 M → 10.4 M params) and **~5× faster**
> (148 ms → 29.5 ms) than the flow expert, is **statistically tied in-distribution** (80.3 vs 79.4 %, Δ=+0.9 pp,
> **p=0.61**), and is **>2× more robust** under distribution shift on the one suite where robustness was measured
> (LIBERO-Spatial, 17.9% → 38.5%, **p<1e-5**).
>
> ⚠️ **Read [`DeepONet PH/CORRECTIONS_2026-06-25.md`](DeepONet%20PH/CORRECTIONS_2026-06-25.md) first** — a statistical
> audit that corrects several earlier overstatements (in-dist is a *tie* not a win; the operator does *not* drive
> accuracy; PH *hurts* robustness; at a fixed budget flow wins the in-dist average; robustness was Spatial-only).

---

## Table of contents
1. [TL;DR — headline results](#tldr--headline-results)
2. [The idea & why it is novel](#the-idea--why-it-is-novel)
3. [Architecture: before vs after](#architecture-before-vs-after)
4. [Component 1 — the DeepONet operator action head](#component-1--the-deeponet-operator-action-head)
5. [Component 2 — the Persistent-Homology (PH) loss](#component-2--the-persistent-homology-ph-loss)
6. [How it is integrated into SmolVLA](#how-it-is-integrated-into-smolvla)
7. [Research papers / references](#research-papers--references)
8. [Results in full](#results-in-full)
9. [Everything that was run — step by step](#everything-that-was-run--step-by-step)
10. [Repository structure — where to find things](#repository-structure--where-to-find-things)
11. [How to reproduce](#how-to-reproduce)
12. [Honest limitations & caveats](#honest-limitations--caveats)
13. [Deep-dive docs](#deep-dive-docs)

---

## TL;DR — headline results

**Model legend:** `M1 = flow` (SmolVLA flow-matching baseline) · `M3 = DeepONet` (operator head) · `M4 = DeepONet + PH loss`.

### LIBERO-Spatial, 5 seeds (the main comparison — `DeepONet PH/v2_results/data/summary.csv`)

| Model | In-dist acc | Robustness (LIBERO-Plus) | Latency | Action-head params |
|---|---|---|---|---|
| **M1 flow** | 79.4 ± 1.4 % | 17.9 ± 5.2 % | 148.1 ms | 99.9 M |
| **M3 DeepONet** | 80.3 ± 2.4 % | **38.5 ± 6.9 %** | **29.5 ms** | **10.4 M** |
| **M4 DeepONet + PH** | **81.5 ± 3.9 %** | 32.6 ± 5.1 % | 29.5 ms | 10.4 M |

- **In-distribution:** DeepONet is a **statistical tie** with flow (Δ=+0.9 pp, **p=0.61** — not significant). Note a
  plain **regression head with no operator scores *higher* in-dist (83.0 %)**, so the in-dist parity comes from the
  cross-attention context, **not** the operator merge — see [ablations](#ablations-libero-spatial).
- **Robustness:** DeepONet is **>2× more robust** than flow and wins **6 of 7** categories — **measured on
  LIBERO-Spatial only** (8.3K, 5 seeds, p<1e-5). No robustness was measured on Object/Long/Goal.
- **Efficiency:** **9.6× fewer** head parameters, **~5× faster** inference (single forward pass vs 10-step flow denoising). *(This is the most robust headline claim.)*
- **PH loss (M4):** its in-dist gain is **not significant** (+1.2 pp, p=0.51) and it **significantly *hurts*
  robustness** (−5.9 pp, **p=0.032**). Recommended config is the **bare operator M3 (PH off)**.

### Per-suite in-distribution (single seed, 15K steps — first comparison)

| Suite | M1 flow | M3 DeepONet | M4 DeepONet+PH |
|---|---|---|---|
| LIBERO-Spatial | 78.5 % | **82.0 %** | 80.5 % |
| LIBERO-Object | 84.5 % | **94.0 %** | 87.0 % |
| LIBERO-Goal | **93.5 %** | 90.0 % | 89.0 % |

### Full-LIBERO sweep at 30K steps + whole-suite average (single seed — `DeepONet PH/v2/deeponet_results/`)

Operator heads re-trained at the **30K-step / batch-48** recipe (matching the flow paper-repro). **At a consistent
30K budget, the flow baseline wins the in-distribution whole-suite average** (Goal exists only at 15K):

| Model | Spatial 30K | Object 30K | Long 30K | Goal 15K* | **4-suite avg** |
|---|---|---|---|---|---|
| **M1 flow (SmolVLA)** | 79.5 | 87.5 | 66.5 | 93.5 | **81.75 %** ✅ |
| M3 DeepONet | 85.0 | 87.0 | 58.5 | 90.0 | **80.12 %** |
| M4 DeepONet+PH | 82.5 | 89.5 | 60.5 | 89.0 | **80.38 %** |

→ **At fixed budget, flow leads the in-dist average (81.75 vs 80.12).** A *previously reported* DeepONet "edge"
(81.88 vs 81.00) only appears under **per-suite best-budget selection** — substituting **Object @ 15K = 94.0** for M3
(its best checkpoint) while flow uses its weaker Object@15K. That selection is not a like-for-like comparison; the
honest headline is the **fixed-budget table above (flow wins in-dist)**. The operator's defensible wins are
**efficiency** and **robustness (Spatial)**, not in-dist accuracy. Full both-ways breakdown:
[`DeepONet PH/runs/report_final/budget_comparison.csv`](DeepONet%20PH/runs/report_final/budget_comparison.csv).
**Long** is where all three are weakest; no model reaches the SmolVLA paper average of **87.3 %**.
*(Goal has no 30K checkpoint — 15K is its only run.)*

### Checkpoint inventory & evaluation protocol

**Three training-step regimes were run** (all SmolVLA: 350 M SmolVLM2 backbone + swapped head):

| Regime | Suites evaluated | Seeds | Metrics | Weights on disk? |
|---|---|---|---|---|
| **8.3K** | Spatial | M1×5, M3×5, M4×5 (+ ablations) | in-dist **+ LIBERO-Plus** | yes (10 ckpts) |
| **15K** | Spatial, Object, Goal | single (seed 0) | in-dist only | **no — deleted** (results kept) |
| **30K** | Spatial, Object, Long | single (seed 0) | in-dist only | yes (9 ckpts) |

- **Only Spatial has all three regimes**; **Long** appears only at 30K, **Goal** only at 15K.
- **Robustness (LIBERO-Plus)** was run **only on Spatial @ 8.3K** (5 seeds). No other suite/regime has robustness.
- **Eval horizons:** in-distribution = **20 episodes/task × 10 tasks, max_steps 520, replan 5**; LIBERO-Plus =
  **15 tasks/category × 7 categories, max_steps 300, replan 5, 1 trial/task**.
- **Significance** (Spatial 5-seed): `DeepONet PH/runs/report_final/significance_tests.csv`;
  **per-seed raw numbers:** `seed_level_spatial_8p3k.csv`; **budget breakdown:** `budget_comparison.csv`.

---

## Scaling to larger flow VLAs — pi0.5 (comp‑1) and GR00T N1.6‑3B (comp‑2)

The operator head won robustness on **SmolVLA (450 M, +20.6 pp, 5‑seed, p<0.001)** and **ACT**. Does it
transfer to much larger flow‑matching / diffusion VLAs? We ran it head‑only on **pi0.5 (3.3 B,
PaliGemma+flow, frozen backbone)** and **GR00T N1.6‑3B (Eagle+diffusion, partly frozen)** — both now complete.
Code: [`DeepONet PH/pi05/`](DeepONet%20PH/pi05/), [`DeepONet PH/comp3_groot/`](DeepONet%20PH/comp3_groot/).

### pi0.5 (comp‑1) — canonical: **photometric‑augmentation** run, replan=5, single seed (`DeepONet PH/pi05/results_aug/`)

Final pi0.5 result: every head trained with identical per‑sample color‑jitter (`--color_jitter`, training‑only) —
the standard way to buy robustness on a **frozen** backbone. All three variants complete, 4/4 suites.

| Suite | Flow in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 87.5 / **71.4** | 92.5 / 44.6 | 90.8 / 35.7 |
| Object | 88.3 / **57.1** | 92.5 / 55.4 | 85.8 / 53.6 |
| Long | 70.0 / **46.4** | 53.3 / 26.8 | 44.2 / 16.1 |
| Goal | 90.8 / **55.4** | 97.5 / 50.0 | 93.3 / 41.1 |
| **Average** | **84.2 / 57.6** | **84.0 / 44.2** | 78.5 / 36.6 |

**Honest result (aug): the factorized head *ties* flow in‑distribution (84.0 vs 84.2) but flow still leads
robustness (+13.4 pp: 57.6 vs 44.2); the shape regularizer (+PH) makes it *worse* (−21.0: 36.6), not better.**
Augmentation lifts both heads (flow Plus 54.9 → 57.6; DeepONet 29.5 → 44.2) and **halves** the gap (−25.4 → −13.4 pp)
but does **not flip its sign** — the SmolVLA/ACT robustness win still does not transfer to a frozen pi0.5 backbone,
and +PH (which *helps* on the adapting backbones) actively hurts here. Real negative result, audited (closed‑loop
replan=5, correct action space / physics‑settle, symmetric flow‑vs‑DeepONet init, matched aggregation, single seed).

**Likely cause — backbone co‑adaptation.** In *both* winning cases the representation was free to adapt to the
operator head (ACT trains the encoder end‑to‑end from scratch; SmolVLA unfroze the backbone in stage 2 at lr 1e‑5).
On pi0.5 the backbone is **frozen and head‑only**, and was pretrained end‑to‑end *as a flow model* — so its
features are native to flow and foreign to a bolted‑on operator head; augmentation regularizes the readout but
cannot let the head co‑adapt the backbone. The signature fits: **in‑dist ties** (the readout still fits) while
**Plus trails** (OOD robustness lives in the representation the head can't reshape). The apples‑to‑apples test
(unfreeze the pi0.5 backbone, matching the SmolVLA regime) is the decisive follow‑up.

#### Provenance — two earlier protocols (superseded, kept for the record)

The aug run is preceded by (1) **replan=5, no augmentation** (`results_replan5/`: flow 84.0 / **54.9**, DeepONet
81.5 / 29.5, +PH 76.3 / 35.7 — gap +25.4 pp) and (2) the original **open‑loop `n_action_steps=50`** run
(`results_openloop_superseded/`: flow 52.9 / 27.7 — artificially low). The replan bug (config default executed all
50 predicted actions open‑loop instead of receding‑horizon replan=5, plus an asymmetric `--init` where the flow
baseline discarded its pretrain) was fixed in `eval_pi05.py`; the campaign was re‑run at replan=5, then re‑run with
augmentation. **The conclusion is invariant across all three protocols:** on a frozen pi0.5 backbone the flow
baseline leads robustness (Plus average — 27.7 vs 16.1 open‑loop; 54.9 vs 29.5 replan=5; **57.6 vs 44.2 with aug**).
Full write‑up + all three tables: [`DeepONet PH/pi05/README.md`](DeepONet%20PH/pi05/README.md).

### GR00T N1.6‑3B (comp‑2) — COMPLETE, single seed (`DeepONet PH/comp3_groot/results_c3/`)

Same head‑only protocol on GR00T (Eagle VLM + diffusion head, partly frozen: the projector and the 1.09 B DiT
train while the Eagle VLM stays mostly frozen). The pipeline required aligning to GR00T's pinned deps
(torch 2.7.1 + flash‑attn 2.8.0.post2) and 7 bring‑up fixes — see
[`DeepONet PH/comp3_groot/README.md`](DeepONet%20PH/comp3_groot/README.md).

| Suite | Diffusion in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 98.3 / **91.1** | 96.7 / 83.9 | 98.3 / 83.9 |
| Object | 98.3 / 83.9 | 99.2 / 75.0 | 99.2 / **82.1** |
| Long | 95.8 / **91.1** | 95.0 / 69.6 | 87.5 / 73.2 |
| Goal | 99.2 / 83.9 | 96.7 / 75.0 | 97.5 / 73.2 |
| **Average** | **97.9 / 87.5** | 96.9 / 75.9 | 95.6 / 78.1 |

**Honest result: the operator head essentially *ties* in‑distribution (96.9 vs 97.9) at a ~100× smaller head
(~11 M vs the 1.09 B DiT) but loses robustness (−11.6 pp: 75.9 vs 87.5); +PH partly recovers it (78.1, −9.4),
mainly by lifting Object‑Plus 75.0→82.1.** This is the *middle* of the co‑adaptation trend: GR00T is only
*partly* frozen, so its robustness loss (−11.6) is smaller than fully‑frozen pi0.5 (−13.4, aug) and far from
adapting SmolVLA (+20.6). The 3 B diffusion baseline is inherently appearance‑robust (Spatial: Light/Sensor/
Texture/Layout/Language all 100; Camera 62.5, Robot‑init 75.0 — an appearance‑robust, geometry‑fragile
signature), so the head has little robustness headroom to add. Single seed; 8 ep/category × 7 = 56 Plus
instances/suite. **No GR00T checkpoints are retained** (the 3 B per‑suite finetunes were pruned after each
confirmed result; the raw per‑category results in `results_c3/` are the durable record).

### ACT — V2 transfer regime (40‑task pretrain → per‑suite finetune) — COMPLETE (`ACT/act_results_v2/`)

On **ACT** (small transformer, trained end‑to‑end — the backbone *does* co‑adapt), the operator head **wins the
whole‑suite average** even under the harder transfer regime (in‑dist **+1.9**, robustness **+3.3** vs baseline),
winning Spatial (+4.3) and Long (+21.0):

| Suite | ACT baseline in / **Plus** | ACT+DeepONet in / **Plus** |
|---|---|---|
| Spatial | 81.0 / 58.3 | **85.3 / 66.7** |
| Object | 81.3 / 45.2 | 65.0 / 45.2 |
| Long | 45.3 / 19.0 | **66.3 / 20.2** |
| Goal | 86.0 / 50.0 | 84.7 / 53.6 |
| **Avg** | 73.4 / 43.1 | **75.3 / 46.4** |

The one exception is **Object (−16.3, a sign‑flip from V1's +5.3)** — root‑caused to **under‑training** under the
short transfer‑finetune (not a bug; V1=30K from‑scratch vs V2=15K pretrain→15K finetune), see
[`ACT/act_results_v2/README.md`](ACT/act_results_v2/README.md). This reinforces the pi0.5 finding: **the operator
head helps when the representation can adapt (ACT, SmolVLA); on a frozen 3.3 B flow backbone (pi0.5) it does not.**

---

## The idea & why it is novel

A VLA model has two parts: a **VLM backbone** (vision + language → token features) and an **action head/expert**
that turns those features into a sequence of robot actions (an "action chunk").

State-of-the-art VLAs (SmolVLA, π0, etc.) use a **flow-matching** or **diffusion** action expert. These are powerful
but **heavy and slow**: they are themselves large transformers, and at inference they require an **iterative denoising
loop** (e.g. 10 forward passes) to produce one action chunk.

**Key observation:** an action chunk is a **function of time** — `a(τ)` for the normalised step `τ ∈ [0, 1]` within the
chunk. Producing a *function* from an *input observation* is exactly the problem that **operator learning** (DeepONet)
was designed for. So instead of denoising noise into actions, we **learn the operator** that maps the current
observation to the action *trajectory* directly.

### The novelty (what is new here)
1. **Operator learning as a VLA action head.** To our knowledge, framing the VLA action chunk as an operator
   `Observation ↦ a(·)` and using a **DeepONet** (Branch ⊗ Trunk) to produce it is a new action-head design for VLAs.
   It replaces the iterative flow/diffusion expert with a **single, parameter-free merge + one forward pass**.
2. **A drop-in, backbone-agnostic head.** The head consumes the generic `(prefix_tokens, pad_mask) → action_chunk`
   interface, so it plugs into *any* VLA backbone (SmolVLA, π0, Octo, GR00T, OpenVLA-OFT, ACT…). Only one small input
   projection rescales with the backbone; the rest is fixed at **~10.4 M params**.
3. **A topological (Persistent-Homology) trajectory loss.** On top of the usual regression loss, we add a **PH loss**
   that matches the *topological shape* of predicted vs ground-truth action trajectories (their persistence diagrams),
   encouraging globally consistent motion rather than only point-wise accuracy.
4. **Robustness as the headline metric.** We show the operator prior is not just smaller/faster but **substantially
   more robust** to real-world distribution shift (camera, lighting, layout, language…), which is where imitation
   policies usually break.

See [`docs/deeponet.md`](docs/deeponet.md) and [`docs/ph_loss.md`](docs/ph_loss.md) for the full theory.

---

## Architecture: before vs after

**Generic pattern (applies to every VLA — only the head changes, the backbone is shared & frozen-ish):**

```
 BEFORE (flow-matching baseline, M1)              AFTER (DeepONet head, M3/M4)
 ┌────────────────────────────┐                  ┌────────────────────────────┐
 │  VLM backbone (SmolVLM2)     │  (unchanged)     │  VLM backbone (SmolVLM2)     │ (unchanged, shared)
 │  ~350 M params               │                  │  ~350 M params               │
 └──────────────┬──────────────┘                  └──────────────┬──────────────┘
                │ prefix tokens [B, N, 960]                       │ prefix tokens [B, N, 960]
                ▼                                                  ▼
 ┌────────────────────────────┐                  ┌──────────────────────────────────────────┐
 │ FLOW-MATCHING ACTION EXPERT  │                 │ DeepONet ACTION HEAD  (10.4 M)             │
 │  ~99.9 M params              │       ⟶         │  CrossAttnPool → Branch(c) ⊗ Trunk(φ(τ))   │
 │  10-step iterative denoise   │      swap       │     → c⊙φ(τ) → OutMLP → a(τ)               │
 │  148 ms / chunk              │                 │  single forward pass · 29.5 ms / chunk     │
 └────────────────────────────┘                  └──────────────────────────────────────────┘
```

### Parameter breakdown (measured, SmolVLA)

| Component (DeepONet head) | Params | Note |
|---|---|---|
| CrossAttnPool (Perceiver-style) | **6.81 M** | `in_proj` 960→512 = 0.49 M · 8 learned queries · 3× cross-attn blocks = 6.31 M |
| Branch network (4096→768→256) | **3.34 M** | dominated by Linear(4096→768) = 3.15 M |
| Trunk network (33→256→256→256) | **0.14 M** | τ + Fourier features → basis φ(τ) |
| Output MLP (256→256→32) | **0.07 M** | + 32 output bias |
| **Total DeepONet head** | **≈ 10.4 M** | vs **99.9 M** flow expert |

| Whole-model comparison | Before (M1 flow) | After (M3/M4 DeepONet) |
|---|---|---|
| Backbone (SmolVLM2, 16 layers) | ~350 M | ~350 M (shared) |
| Action head | 99.9 M | **10.4 M** |
| **Total** | **~450 M** | **~360 M** (−20% if dead expert removed) |
| Inference (action chunk) | 148 ms | **29.5 ms** (~5×) |

> The same **~10.4 M head** drops into much larger backbones (π0 ~3.3 B, OpenVLA ~7 B, GR00T ~3 B); only the input
> projection rescales (≈ +0.5–1.5 M). The bigger the original action expert it replaces, the larger the saving.
> Full cross-model table in [`docs/architecture.md`](docs/architecture.md).

---

## Component 1 — the DeepONet operator action head

**File:** [`DeepONet PH/v2/deeponet_head_v2.py`](DeepONet%20PH/v2/deeponet_head_v2.py)

A **DeepONet** (Deep Operator Network, Lu et al. 2021) approximates an operator `G` mapping an input function/vector `u`
to an output function `G(u)(y)` via two sub-networks whose outputs are merged by a dot product:

```
G(u)(y) ≈ Σ_{k=1..p}  b_k(u) · t_k(y)      (Branch · Trunk)
```

Here we instantiate it as the action head:

1. **CrossAttnPool** — Perceiver-style cross-attention. `K = 8` learned query tokens attend over the full VLM prefix
   (`d_model = 512`, 8 heads, 3 blocks) → a fixed-size summary of the observation, regardless of prefix length.
2. **Branch network** — maps the pooled observation → `p = 256` coefficients `c ∈ ℝ²⁵⁶` (the "what to do" code).
3. **Trunk network** — maps the query coordinate `τ ∈ [0,1]` (normalised time within the action chunk), lifted by
   **Fourier features**, → `p = 256` basis functions `φ(τ) ∈ ℝ²⁵⁶` (the "shape over time").
4. **Parameter-free merge** — `c ⊙ φ(τ)` (element-wise), then a small **Output MLP** → the action at time `τ`.

```
a(τ) = OutMLP( Branch(obs) ⊙ Trunk(τ) )
```

Evaluating the trunk at `τ = 0, 1/T, …, 1` yields the **whole action chunk in one forward pass** — no denoising loop.

**Why `p = 256`?** It is the number of **basis functions** (operator latent dimension), *not* a time discretisation.
The chunk can be queried at any `τ` (continuous in time); `p` controls how rich the learned function space is.
`d_model = 512` is the internal transformer width of the cross-attention pooler. See [`docs/deeponet.md`](docs/deeponet.md).

---

## Component 2 — the Persistent-Homology (PH) loss

**File:** [`DeepONet PH/v2/ph_loss.py`](DeepONet%20PH/v2/ph_loss.py)

Standard training uses an MSE/L1 **point-wise** regression loss on the action chunk. The **PH loss** adds a
**topological** term: it computes the **persistence diagram** of the predicted and ground-truth action trajectories
(via a Vietoris–Rips / sublevel-set filtration) and penalises the difference between them.

Intuitively, PH captures the **global shape** of a trajectory — its loops, connected components, and how long
topological features "persist" across scales — which point-wise losses ignore. Matching it encourages the policy to
produce trajectories that are *globally* the right shape, not just locally close.

- `M3` = DeepONet head, **PH loss off**.
- `M4` = DeepONet head, **PH loss on**.

**Empirically (corrected — see [`CORRECTIONS_2026-06-25.md`](CORRECTIONS_2026-06-25.md)):** PH's in-distribution
"gain" (M4 81.5 % vs M3 80.3 %, +1.2 pp) is **not statistically significant** (t-p = 0.51), and PH **significantly
*hurts* robustness** (M4 32.6 % vs M3 38.5 %, **−5.9 pp, t-p = 0.032**). So PH is **not recommended** — the headline
operator model is **M3 (PH off)**. See [`docs/ph_loss.md`](docs/ph_loss.md).

---

## How it is integrated into SmolVLA

**File:** [`DeepONet PH/v2/modeling_smolvla_deeponet_v2.py`](DeepONet%20PH/v2/modeling_smolvla_deeponet_v2.py)
(class `SmolVLADeepONetPolicy`).

SmolVLA = **SmolVLM2-500M** backbone (reduced to 16 layers, ~350 M) + a flow-matching action expert (~99.9 M).
The integration:

1. **Keep** the SmolVLM2 backbone and its prefix-token construction unchanged.
2. **Swap** the flow-matching expert for the DeepONet head (the generic `prefix_tokens [B,N,960], pad_mask → action_chunk [B,T,32]` interface).
3. **Load** the pretrained SmolVLA checkpoint with `strict=False`, then **freeze the dead flow modules**
   (`lm_expert`, `action_in_proj/out_proj`, `action_time_mlp`) so only the backbone + new head train.
4. **Two-stage fine-tune:** stage-1 head-only warm-up (1650 steps), then stage-2 full fine-tune; EMA on weights.

The head is the **generic plug-in**; `SmolVLADeepONetPolicy` is just the SmolVLA-specific adapter. Porting to π0 is
documented in [`DeepONet PH/PORTING_TO_PI0.md`](DeepONet%20PH/PORTING_TO_PI0.md). Full integration notes in
[`docs/smolvla_integration.md`](docs/smolvla_integration.md).

---

## Research papers / references

| Topic | Paper | Used for |
|---|---|---|
| **DeepONet** | Lu, Jin, Pang, Zhang, Karniadakis — *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, **Nature Machine Intelligence 2021** ([arXiv:1910.03193](https://arxiv.org/abs/1910.03193)) | the operator action head (Branch ⊗ Trunk) |
| **SmolVLA** | Hugging Face / LeRobot — *SmolVLA: A Vision-Language-Action model for affordable and efficient robotics* (2025) ([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) | the backbone & flow-matching baseline |
| **Flow matching** | Lipman, Chen, Ben-Hamu, Nickel, Le — *Flow Matching for Generative Modeling*, **ICLR 2023** ([arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) | the baseline action expert it replaces |
| **Persistent Homology (TDA)** | Edelsbrunner & Harer — *Computational Topology: An Introduction* (2010); Carlsson — *Topology and data*, **Bull. AMS 2009** | the topological PH loss |
| **Topological / PH loss in deep learning** | Hu, Li, Samaras, Chen — *Topology-Preserving Deep Image Segmentation*, **NeurIPS 2019** ([arXiv:1906.05404](https://arxiv.org/abs/1906.05404)) | differentiable persistence-diagram loss |
| **LIBERO benchmark** | Liu et al. — *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, **NeurIPS 2023** ([arXiv:2306.03310](https://arxiv.org/abs/2306.03310)) | evaluation suites (Spatial/Object/Goal/Long) |
| **Fourier features** | Tancik et al. — *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, **NeurIPS 2020** ([arXiv:2006.10739](https://arxiv.org/abs/2006.10739)) | the Trunk's τ encoding |

*(Citations are the standard references for each method; verify exact venue/year against the linked source before formal use.)*

---

## Results in full

### LIBERO-Plus robustness by perturbation (Spatial only, `DeepONet PH/v2_results/data/robustness_per_category.csv`)

> **Scope:** LIBERO-Plus robustness was measured on **LIBERO-Spatial only** (8.3K, 5 seeds). There are **no**
> robustness numbers for Object / Long / Goal, so the ">2× more robust" headline is a **single-suite** result.

| Perturbation | M1 flow | M3 DeepONet | M4 DeepONet+PH |
|---|---|---|---|
| Camera Viewpoints | 18.7 | **26.7** | 14.7 |
| Light Conditions | 22.7 | **61.3** | 54.7 |
| Sensor Noise | **25.3** | 21.3 | 17.3 |
| Background Textures | 24.0 | 42.7 | **44.0** |
| Objects Layout | 9.3 | **48.0** | 46.7 |
| Robot Initial States | 17.3 | **36.0** | 29.3 |
| Language Instructions | 8.0 | **33.3** | 21.3 |
| **Average** | **17.9** | **38.5** | **32.6** |

→ DeepONet (M3) wins **6/7** categories; flow wins only Sensor Noise. The largest gains are on **Objects Layout**
(9.3 → 48.0) and **Language** (8.0 → 33.3) — exactly the shifts that break imitation policies.

### Ablations (LIBERO-Spatial — `DeepONet PH/Ablation_Results/ablations.csv`)

| Configuration | In-dist | Robustness | Seeds |
|---|---|---|---|
| **Full DeepONet-v2 (p256, 3 blocks, Fourier)** | 80.3 ± 2.4 | **38.5 ± 6.9** | 5 |
| (−) basis p: 256→64 | 81.7 ± 0.5 | 32.7 ± 5.2 | 3 |
| (−) Fourier-τ (linear only) | 82.7 ± 1.4 | 39.0 ± 8.7 | 3 |
| (−) cross-attn blocks: 3→1 | 78.2 ± 2.0 | 30.8 ± 4.5 | 3 |
| Regression head (NO operator, same context) | 83.0 ± 4.2 | 32.7 ± 0.9 | 3 |

**Reading the ablations:** the **cross-attention pooler** matters most (3→1 blocks hurts both metrics); the
**operator merge** is what buys robustness — a plain regression head with the same context gets higher in-dist (83.0)
but loses ~6 points of robustness (32.7 vs 38.5). `p` and Fourier trade a little in-dist for robustness/stability.

Full numbers, plots, and per-task tables: [`docs/results.md`](docs/results.md).

### 30K paper-reproduction flow campaign — COMPLETE (batch 48 — `DeepONet PH/paper_repro/`)

Higher-budget flow-matching runs (30,000 steps, batch 48) to push the flow baseline toward its published
LIBERO values. **Finished 2026-06-17** (in-distribution, 20 ep/task, replan 5):

| Suite | Flow 10-task (official) | Flow 9-task (excl-lowest) |
|---|---|---|
| LIBERO-Spatial | 79.5 % | 87.78 % (excl task5) |
| LIBERO-Object | 87.5 % | 90.56 % (excl task9) |
| LIBERO-Long (LIBERO-10) | 66.5 % | 70.56 % (excl task0) |

These came in **below the published targets** (Spatial 90 / Object 96 / Long ≥71). The matching
**DeepONet (m3) and DeepONet+PH (m4) 30K runs** are packaged for a budget-matched comparison on a second GPU
— see the self-contained **`transfere/`** bundle (`transfere.zip`, `SETUP_AND_RUN.md`, `run_deeponet_m3m4.sh`).

### Goal object-layout / anti-memorisation test — COMPLETE (`DeepONet PH/goal_layouttest/`)

The frozen LIBERO-Goal flow model re-evaluated over **3 different object-position layout slices** covering all
50 init-states, 20 ep/task. Accuracy holds across unseen layouts → **no memorisation**:

| Layout slice | init-states | seed | Goal avg |
|---|---|---|---|
| off_0 (canonical) | [0:20] | 1000 | 91.5 % |
| off_20 | [20:40] | 1020 | 88.0 % |
| off_30 | [30:50] | 1030 | 91.5 % |
| **Mean ± std** | — | — | **90.3 ± 1.7 %** |

### DeepONet (m3) + DeepONet+PH (m4) at 30K steps — full LIBERO sweep — COMPLETE (`DeepONet PH/v2/deeponet_results/`)

The operator heads were trained at the **same 30K-step / batch-48 recipe as the flow baseline** on Spatial,
Object, and Long (single seed = 0), each then evaluated in-distribution (20 episodes/task, replan 5) with
per-task plots, per-episode videos, and a flow-comparison plot. **Finished 2026-06-21.**

**Step by step (what the campaign did):** `run_deeponet_m3m4.sh` →
1. For each suite (Spatial → Object → Long): train **m3** (`--head deeponet --variant baseline`), then **m4**
   (`--head deeponet --variant ph --lambda_ph 0.02 --ph_k 8`), both 30K (stage1 1650 + stage2 28350), batch 48,
   EMA 0.999, from `lerobot/smolvla_base`.
2. Evaluate each model in-distribution (20 ep/task, replan 5) → `runs/eval_{m3,m4}/`.
3. `summarize_suite.py` → full 10-task + 9-task (excl-lowest) averages; `plot_10_and_9.py` → per-model bar plots.
4. `compare_vs_flow.py --suite <S>` → grouped per-task plot **m3 vs m4 vs flow** + BEAT/below verdict.

**Per-suite in-distribution results (10-task average %):**

| Suite (budget) | M1 flow | M3 DeepONet | M4 DeepONet+PH | winner |
|---|---|---|---|---|
| Spatial (30K) | 79.5 | **85.0** | 82.5 | m3 (+5.5) |
| Object (30K) | 87.5 | 87.0 | **89.5** | m4 (+2.0) |
| Object (15K)\* | 84.5 | **94.0** | 87.0 | m3 (+9.5) |
| Long (30K) | **66.5** | 58.5 | 60.5 | flow (+6–8) |
| Goal (15K) | **93.5** | 90.0 | 89.0 | flow |

**\* Object overfits at 30K** — see findings below; its 15K checkpoint (94.0) is the better model.

**Final whole-LIBERO average** — each suite at its **non-overfit best-budget checkpoint** (Object & Goal @15K,
Spatial & Long @30K):

| Model | Spatial (30K) | Object (15K) | Long (30K) | Goal (15K) | **4-suite average** |
|---|---|---|---|---|---|
| M1 flow (SmolVLA) | 79.5 | 84.5 | 66.5 | 93.5 | **81.00 %** |
| **M3 DeepONet** | 85.0 | 94.0 | 58.5 | 90.0 | **81.88 %** ✅ |
| M4 DeepONet+PH | 82.5 | 87.0 | 60.5 | 89.0 | **79.75 %** |

→ At each suite's best-budget checkpoint, **M3 DeepONet edges out the SmolVLA-flow baseline on the whole-suite
average (81.88 vs 81.00)** — while being **~9.6× smaller, ~5× faster, and ~2× more robust** (per the ablations
above). At a single fixed 30K budget the three are within ~1.6 pt (flow 81.75 / m4 80.38 / m3 80.12). No model
reaches the original SmolVLA paper average of 87.3 % — **Long is the bottleneck** (~58–66 % for all three).

**Key findings:**
- **Object over-trains at 30K** (94.0 → 87.0, a broad −7 across 9/10 tasks; train-MSE fell to ~0.027). Object
  saturates by ~15K, so extra steps overfit it. *Evidence it's not a bug:* Spatial **improved** with 30K (82→85).
- **Long is genuinely hard, not overfit** — its train-MSE stayed **highest** (0.041 vs ~0.025), and its per-task
  scores are **bimodal** (some tasks ~100 %, some ~10–20 %). It's long-horizon error-compounding, where the
  single-pass operator head is weaker than flow's iterative denoising (flow 66.5 > m3 58.5 / m4 60.5).
- **PH (m4) helps Object** (+2.0 over flow, best there) but trails m3 elsewhere — consistent with PH being an
  accuracy regulariser while robustness comes from the operator structure.

Comparison plots & per-task tables: `DeepONet PH/v2/deeponet_results/<Suite>/plots/<Suite>_compare_vs_flow.png`
(+`.csv`); per-episode videos in `runs/eval_{m3,m4}/episode_videos/`.

---

## Everything that was run — step by step

A complete chronological log is in [`docs/experiments_log.md`](docs/experiments_log.md). Summary:

1. **Weeks 1–3 — Flow-matching baseline (SmolVLA).** Set up SmolVLA + LeRobot, fine-tuned the flow baseline on LIBERO,
   established the evaluation harness. → `Flowmatching PH (Weeks 1-3)/`
2. **DeepONet v1.** First operator head. Matched flow in-dist but **collapsed on some robustness categories**
   (camera/sensor) → motivated v2. → `DeepONet PH/v1_results/`
3. **DeepONet v2 (main).** Redesigned head (CrossAttnPool + p256 Branch⊗Trunk + Fourier). **5-seed** head-to-head of
   M1/M3/M4 on Spatial for in-dist + LIBERO-Plus robustness + latency + params. → `DeepONet PH/v2_results/`
4. **Ablations.** p256→64, no-Fourier, 3→1 cross-attn blocks, regression-head (no operator). → `DeepONet PH/Ablation_Results/`
5. **Per-suite generalisation.** Trained/evaluated M1/M3/M4 on **Object** and **Goal** (and Spatial) at 15K steps,
   with per-task plots + per-episode videos. → `DeepONet PH/{Spatial,Object,Goal}/`
6. **Paper-reproduction campaign (done, 2026-06-17).** 30K-step, batch-48 flow runs. Final flow in-dist:
   **Spatial 79.5 % · Object 87.5 % · Long 66.5 %** (10-task) — below the paper targets (90 / 96 / ≥71). The
   budget-matched **DeepONet m3/m4 30K** runs are bundled in `transfere/` for a second GPU. → `DeepONet PH/paper_repro/`
7. **task5 investigation.** Why LIBERO-Spatial task5 (stacked-bowl grasp) is a hard ceiling; replan sweep (inference
   tuning did not fix it). → `DeepONet PH/v2/task5_fix_experiment.py`
8. **Goal object-layout test (done).** Generalisation/anti-memorisation check on 3 object-layout slices (all 50
   init-states): **90.3 ± 1.7 %** (91.5 / 88.0 / 91.5) — holds across unseen layouts → no memorisation. → `DeepONet PH/goal_layouttest/`
9. **Pi0 porting analysis.** Time/feasibility of porting the head to π0. → `DeepONet PH/PORTING_TO_PI0.md`
10. **DeepONet 30K campaign (done, 2026-06-21).** Trained **m3 + m4** at the full 30K-step / batch-48 recipe on
    Spatial/Object/Long (single seed), evaluated in-distribution + per-task plots + per-episode videos +
    **m3/m4/flow comparison plots**. Result: m3 edges flow on the whole-suite average (**81.9 vs 81.0**); Object
    over-trains at 30K (15K is its better checkpoint), Long is the hard long-horizon suite. →
    `DeepONet PH/v2/deeponet_results/` + `DeepONet PH/v2/compare_vs_flow.py`

---

## Repository structure — where to find things

```
.
├── README.md                         ← you are here (master overview)
├── docs/                             ← deep-dive documentation
│   ├── architecture.md               ← before/after diagrams + cross-model param tables
│   ├── deeponet.md                   ← the DeepONet operator head, in depth
│   ├── ph_loss.md                    ← the persistent-homology loss, in depth
│   ├── smolvla_integration.md        ← how the head plugs into SmolVLA
│   ├── experiments_log.md            ← full chronological log of everything run
│   └── results.md                    ← all result tables (in-dist, robustness, ablations, per-task)
│
├── DeepONet PH/                      ← MAIN project
│   ├── v2/                           ← v2 source (the current/best version)
│   │   ├── deeponet_head_v2.py       ← the DeepONet operator head
│   │   ├── modeling_smolvla_deeponet_v2.py  ← SmolVLA adapter (swaps the head)
│   │   ├── modeling_smolvla_ph.py    ← flow baseline policy (+PH hooks)
│   │   ├── ph_loss.py                ← persistent-homology loss
│   │   ├── regression_head.py        ← ablation: plain regression head
│   │   ├── train.py / evaluate.py / evaluate_plus.py   ← training & eval
│   │   ├── libero_v_wrapper.py / libero_plus_wrapper.py ← env wrappers
│   │   ├── make_suite_plots.py / make_videos*.py        ← plots & videos
│   │   ├── paper_repro_30k.sh        ← the 30K paper-repro flow campaign
│   │   ├── run_deeponet_m3m4.sh + compare_vs_flow.py  ← DeepONet m3/m4 30K campaign + flow comparison
│   │   ├── deeponet_results/         ← m3/m4 30K outputs: runs/, eval_{m3,m4}/, plots/, videos, PROGRESS.log (done)
│   │   ├── run_goal_seedtest.sh + evaluate_seedtest.py  ← Goal object-layout test
│   │   └── *.sh                      ← all run/orchestration scripts
│   ├── Spatial/ Object/ Goal/        ← per-suite 15K results: runs/ (eval JSONs), plots/, logs/, videos/
│   ├── v1_results/ v2_results/       ← packaged results + plots + CSV summaries
│   ├── Ablation_Results/             ← ablation JSONs, CSV, plot
│   ├── DeepONet_Results/             ← final report + collated data
│   ├── paper_repro/                  ← 30K flow campaign outputs + PROGRESS.log  (done)
│   ├── goal_layouttest/              ← Goal object-layout test outputs       (done)
│   └── PORTING_TO_PI0.md             ← π0 porting analysis
│
├── transfere/ + transfere.zip        ← self-contained bundle to train DeepONet m3/m4 (30K) on a 2nd GPU
└── Flowmatching PH (Weeks 1-3)/      ← early flow-matching baseline work + presentation
```

**Quick find:**
- 📊 **Plots** → `DeepONet PH/*/plots/*.png`, `DeepONet PH/{v2_results,Ablation_Results}/*.png`
- 📈 **Result tables (CSV/JSON)** → `DeepONet PH/v2_results/data/`, `*/runs/eval_*/success_rates.json`
- 🎬 **Videos** → `DeepONet PH/*/runs/eval_*/episode_videos/`, `DeepONet PH/comparison_videos/`
- 📝 **Logs** → `DeepONet PH/*/logs/`, `DeepONet PH/paper_repro/PROGRESS.log`

> **Not included in this repo (too large for GitHub):** model checkpoints (`*.safetensors`, ~40 GB total), the
> Python `venv/`, `third_party/` libraries, and HF dataset caches. **Checkpoints are backed up on Hugging Face** so
> the study is fully recoverable from GitHub + HF:
> - SmolVLA flow / DeepONet m3·m4 + ablations → [`AyushShah1107/deeponet-ph-vla-checkpoints`](https://huggingface.co/AyushShah1107/deeponet-ph-vla-checkpoints)
> - ACT-backbone campaign (ACT / +DeepONet / +PH) → [`AyushShah1107/act-deeponet-libero-checkpoints`](https://huggingface.co/AyushShah1107/act-deeponet-libero-checkpoints)
>
> Rebuild the env with `pip install -r requirements.txt`; re-clone `third_party/` (LIBERO-plus `sylvestf/LIBERO-plus`,
> LIBERO `Lifelong-Robot-Learning/LIBERO`). Everything else (code, results, plots, videos, logs) **is** in this repo.
> The ACT campaign — including LIBERO-Plus robustness, latency, and plots — is documented in [`ACT/README.md`](ACT/README.md).

---

## How to reproduce

```bash
# 1. Environment (Python 3.12) — install LeRobot + SmolVLA deps + LIBERO + persim/ripser for PH
pip install -r requirements.txt        # (see docs; uses lerobot, robosuite, libero, torch, persim)
export MUJOCO_GL=egl

# 2. Train one model (head ∈ {flow, deeponet}; variant ∈ {baseline, ph})
cd "DeepONet PH/v2"
python train.py --head deeponet --variant baseline \
  --dataset lerobot/libero_spatial_image --out runs/m3_s0 \
  --stage1_steps 1650 --stage2_steps 28350 --stage1_batch 48 --stage2_batch 48

# 3. Evaluate in-distribution + LIBERO-Plus robustness
python evaluate.py      --suite libero_spatial --model m3=deeponet=runs/m3_s0/checkpoints/LATEST --only indist
python evaluate_plus.py --suite libero_spatial --model m3=deeponet=runs/m3_s0/checkpoints/LATEST

# 4. Plots
python make_suite_plots.py --suite_dir ../Spatial --suite libero_spatial --label Spatial
```

---

## Honest limitations & caveats
> Full statistical audit with p-values and backing CSVs:
> [`DeepONet PH/CORRECTIONS_2026-06-25.md`](DeepONet%20PH/CORRECTIONS_2026-06-25.md).
- **In-distribution accuracy is a statistical tie** (M3 vs flow Δ=+0.9 pp, **p=0.61**), and the parity does **not**
  come from the operator — a plain regression head with no operator scores *higher* in-dist (83.0 %). The operator's
  measurable contribution is robustness, not accuracy.
- **At a fixed 30K budget, flow wins the in-dist whole-suite average** (81.75 vs 80.12). The DeepONet "edge" only
  appears under per-suite best-budget selection (Object@15K). See `runs/report_final/budget_comparison.csv`.
- **Robustness was measured on LIBERO-Spatial only** (8.3K, 5 seeds). No Object/Long/Goal robustness exists — the
  ">2× more robust" claim is single-suite.
- **PH loss is not recommended:** no significant in-dist gain (p=0.51) and a **significant robustness loss**
  (−5.9 pp, p=0.032). Use the bare operator **M3**.
- **In-dist numbers are imitation (behavioral cloning)** success rates; high in-dist does *not* by itself prove
  generalisation — the LIBERO-Plus robustness numbers are the distribution-shift test.
- **Per-suite 15K/30K numbers are single-seed** (no error bars); only Spatial @ 8.3K has 5 seeds + significance.
  Treat single-seed numbers as ± a few % sampling noise.
- **15K checkpoints were deleted** (disk pressure) — only their eval JSONs remain; 8.3K and 30K weights are on disk.
- The 30K head-to-head **is now complete** (`DeepONet PH/v2/deeponet_results/`, in-dist, single seed).
- Cross-model parameter savings for π0/Octo/GR00T/OpenVLA in `docs/architecture.md` use **published model sizes**
  (approximate); only the SmolVLA numbers are measured here.

---

## Deep-dive docs
- [`docs/architecture.md`](docs/architecture.md) — before/after diagrams, full param breakdown, cross-model porting table
- [`docs/deeponet.md`](docs/deeponet.md) — operator learning & the DeepONet head, in depth
- [`docs/ph_loss.md`](docs/ph_loss.md) — persistent homology & the topological loss
- [`docs/smolvla_integration.md`](docs/smolvla_integration.md) — SmolVLA adapter internals
- [`docs/experiments_log.md`](docs/experiments_log.md) — full step-by-step of everything run
- [`docs/results.md`](docs/results.md) — every result table

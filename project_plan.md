# Project: Representation Geometry of Safety Failure
### Mapping how unsafe capabilities survive RepE safety defenses
**Compute target:** Google Colab free tier, T4 GPU (16GB VRAM), models ≤3–9B in 4-bit/8-bit.

Core claim being tested: *RepE safety defenses suppress unsafe behavior, but the unsafe
representation often remains linearly recoverable and re-elicitable through nearby
steering directions, and this pattern generalizes across defense types.*

**Model set (updated):** Gemma-2-2B-IT, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct,
Phi-4-mini-instruct (~3.8B). Llama-3.2-3B was already in the original plan; Phi-4-mini
is the new addition, chosen because it's a different architecture/training lineage from
the other three, which strengthens the "does this generalize across models" claim.

---

## Stage 0 — Environment & Repo Setup
- Create repo structure:
  ```
  /data          # harmful/harmless prompt pairs, held-out test sets
  /models        # cached HF checkpoints, circuit-breaker weights
  /src
    extraction.py    # activation extraction hooks (reuse hackathon pipeline)
    probes.py        # linear probe training/eval, layer sweep
    steering.py       # steering vector construction (diff-of-means, CAA-style)
    defenses.py       # apply refusal-ablation, circuit-breaker defense
    recovery.py        # re-elicitation attacks, cross-method recovery
    metrics.py         # recovery strength, decodability score, elicitation score
  /notebooks     # Colab-ready experiment notebooks, one per stage
  /results       # saved probe accuracies, steering magnitudes, plots
  /paper         # writeup, figures
  ```
- Pin package versions: `transformers`, `accelerate`, `bitsandbytes`, `torch`, `scikit-learn`, `matplotlib`, `datasets`.
- Load and smoke-test all 4 base models in 4-bit on T4:
  - Gemma-2-2B-IT
  - Llama-3.2-3B-Instruct
  - Qwen2.5-3B-Instruct
  - Phi-4-mini-instruct
- **Deliverable:** repo scaffold + all 4 models loading and generating on T4 within memory budget.

---

## Stage 1 — Data Construction
- Build/reuse contrastive prompt sets: harmful vs. harmless pairs (reuse refusal-steering
  hackathon dataset as base; extend with AdvBench / harmful-behaviors subset for volume).
- Split into: **probe-train**, **probe-test** (held out), **steering-construction set**,
  **recovery-evaluation set** (never seen during probe/steering fitting — critical for
  a clean claim).
- Also prepare a **benign-but-adjacent** prompt set for the counterfactual control
  (Stage 6) — prompts that sound risky but aren't (e.g., safety-research phrasing).
- **Deliverable:** versioned dataset splits saved to `/data`, with a short data card.

---

## Stage 2 — Baseline Activation Extraction & Refusal Behavior
- For each of the 4 base (undefended) models, extract residual-stream activations at
  every layer for harmful and harmless prompts.
- Confirm baseline refusal behavior on harmful prompts (sanity check: models refuse
  before any defense is applied).
- **Deliverable:** cached activation tensors per model per layer; baseline refusal rate table.

---

## Stage 3 — Apply RepE Defenses (two families)
- **Defense A — Refusal-direction ablation / steering-based suppression**
  Build the refusal direction (diff-of-means, Arditi et al. style) and ablate/suppress it
  at inference time. Fully buildable from scratch on T4.
- **Defense B — Circuit breakers**
  Prefer using an existing public circuit-breaker checkpoint/implementation
  (Zou et al., "Improving Alignment and Robustness with Circuit Breakers") if a
  compatible open-weight checkpoint exists for any of the 4 models; otherwise
  implement a lightweight circuit-breaker-style loss on a small LoRA adapter
  (scope down to 1–2 models if full retraining is too costly on T4 — see compute
  budget note below, since adding a 4th model raises this risk).
- Verify each defense actually suppresses harmful outputs on the recovery-evaluation
  set (behavioral check: refusal/compliance rate before vs. after).
- **Deliverable:** 2 defended model variants per base model (or 1 full + 1 scoped, per
  compute reality), with confirmed suppression on held-out harmful prompts.

---

## Stage 4 — Layer-Wise Survival Map (Probe Sweep)
- Train a linear probe at **every layer** of each defended model to classify
  harmful vs. harmless activations.
- Plot probe accuracy vs. layer index, for each model × each defense.
- Identify: does the unsafe concept get pushed out of the representation, or does it
  persist at similar accuracy in one or more layers despite the behavioral suppression?
- **Deliverable:** layer-wise accuracy curves (the core "survival map" figure) for all
  model × defense combinations.

---

## Stage 5 — Recovery Strength & Cross-Method Recovery
- **Same-method recovery:** for each defense, try to re-elicit harmful behavior using a
  steering vector built the same way the defense's own concept was represented.
  Measure recovery strength: minimum steering magnitude (alpha) needed to flip behavior,
  and resulting probe-decodability at that point.
- **Cross-method recovery:** defend with method A, attempt recovery with method B's
  steering approach (e.g., circuit-breaker-defended model, refusal-direction-style
  recovery vector). This is the key novel comparison — same-representation evidence.
- Log: recovery success rate, steering magnitude required, generation quality
  (fluency/coherence check) at the recovery point, to catch cases where recovery
  degrades output too much to count as "real" recovery.
- **Deliverable:** recovery strength table (model × defense × recovery-method), with
  cross-method recovery highlighted separately.

---

## Stage 6 — Counterfactual Control (Decodable vs. Elicitable vs. Generated)
- Apply a **benign** steering direction (unrelated concept, e.g., formality or verbosity)
  to a defended model and confirm:
  - the unsafe concept is *still decodable* by the layer-wise probe (Stage 4 result holds)
  - but *not elicitable* by this benign steering — behavior stays safe
- This separates three distinct properties cleanly:
  1. **Decodable** — probe finds it
  2. **Elicitable** — targeted steering brings the behavior back
  3. **Generated** — harmful text actually comes out, verified by output classifier
- **Deliverable:** a 3-way table (decodable / elicitable / generated) per model × defense,
  the paper's central scientific contribution.

---

## Stage 7 — Analysis & Figures
- Aggregate across all 4 models to check generality of the survival pattern
  (not model-specific). With 4 models spanning different architectures/lineages,
  this is a stronger generality claim than the original 3-model plan.
- Key figures:
  1. Layer-wise survival curves (all 4 models overlaid, per defense)
  2. Recovery strength bar chart (same-method vs. cross-method)
  3. Decodable/elicitable/generated 3-way breakdown
- Statistical check: is the survival effect consistent across models, or does it vary
  meaningfully (report both, don't overclaim generality if it doesn't hold).
- **Deliverable:** final figure set + results tables ready for the paper.

---

## Stage 8 — Writeup
- Draft in this order: Method (Stages 3–6 as the pipeline) → Results (Stage 7 figures)
  → Related work (position clearly against unlearning-reversibility literature,
  RepE survey, obfuscated-activations work) → Discussion (what this implies for
  RepE as a safety technique) → Limitations (T4-scale models only, 2 defense families,
  scope of circuit-breaker implementation, and note explicitly if circuit-breaker
  had to be scoped to fewer than all 4 models due to compute).
- Target venue framing: main-track submission (ICML/ICLR/NeurIPS) with
  SaTML/AIES/FAccT as backup; also usable directly as a fellowship work sample
  (MATS/SPAR/Apart/Astra) even if only Stages 0–6 are complete.

---

## Compute Budget Notes for Claude Code
- Keep all activation caches on disk (not in-memory) between stages — Colab sessions
  disconnect; extraction (Stage 2) should be resumable/checkpointed.
- 4-bit quantization for all base models; avoid loading more than one model at a time.
- **Going from 3 to 4 models roughly adds 33% to every stage's runtime** (extraction,
  probe sweeps, steering, recovery). This is fine for probes/steering (cheap), but
  raises the risk level for circuit-breaker training specifically.
- Circuit-breaker training (if done from scratch) is the single biggest compute risk —
  scope to 1 model first (recommend Llama-3.2-3B or Qwen2.5-3B, both have more public
  reference implementations to build from), expand to a 2nd model only if time/compute
  allows. Do not attempt from-scratch circuit-breaker training on all 4 models.
- Refusal-direction ablation (Defense A) is cheap and should be run on all 4 models
  without concern — this is where the 4-model generality claim mainly comes from.
- Probe training (Stage 4) and steering vector construction (Stage 5) are cheap
  (seconds–minutes each) — safe to run exhaustively across layers for all 4 models.
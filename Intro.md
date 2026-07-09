**Project One-Pager: Mapping How Unsafe Capabilities Survive RepE Safety Defenses**

**Working title:** Representation Geometry of Safety Failure: Where Unsafe Capabilities Hide After RepE Defenses

**Why it matters**
RepE defenses (steering vectors, circuit breakers, refusal directions) are sold as lightweight, elegant safety fixes for LLMs. But nobody has carefully checked whether these defenses actually erase the unsafe capability, or just make it harder to trigger while it's still sitting inside the model. If it's the second one, that's a false sense of security for anyone deploying these techniques.

**The gap**
Prior work on "unlearning reversibility" checks *whether* a fact/behavior can be recovered — a yes/no answer, using gradient-based fine-tuning attacks. Nobody has mapped *how* and *where* a suppressed unsafe capability survives inside a RepE-defended model, or whether this pattern holds across different RepE defense types.

**What we propose**
Move from a single yes/no recovery check to a **mechanism map**: which layers still hold the unsafe concept, how strong the leftover signal is, and whether it can be pulled back out using directions *nearby* to (not identical to) what the defense removed. Core claim we're testing: *RepE defenses suppress behavior, but the unsafe representation often remains linearly recoverable and re-elicitable through nearby steering directions.*

**How we're doing it** (all reuses your existing probe/steering pipeline — no extra heavy compute)
1. **Layer-wise probe sweep** — instead of one probe at one layer, train a probe at every layer to see exactly where the unsafe concept is still linearly readable after defense. Cheap, just more probes on activations you already have.
2. **Two defense families, not one** — compare refusal-direction ablation (steering-based) vs. circuit breakers (using existing published circuit-breaker checkpoints where possible, to avoid retraining cost). If the same survival pattern shows up in both, it's a general RepE claim, not a one-method quirk.
3. **Recovery strength, not just success** — measure *how much* steering magnitude is needed to bring the behavior back, and how much probe accuracy survives. Turns "recovered / not recovered" into a graded score.
4. **Cross-method recovery** — defend with method A (e.g., circuit breaker), try to recover with method B (e.g., steering vector). If recovery works across methods, that points to a shared underlying representation, which is a much stronger finding.
5. **Counterfactual control** — apply a *benign* steering direction and confirm the unsafe concept is still decodable by a probe but not actually triggerable. This separates three distinct things cleanly: **decodable** (probe can find it) vs. **elicitable** (steering brings it back) vs. **actually generated** (harmful text comes out).

**Resource fit**
All additions are extra *conditions* on the same activation-extraction/probe/steering pipeline you already built — not new heavy training. The one place we scope down for T4: circuit breakers use existing public checkpoints rather than training our own, and we cap the study at 2 defense families and 3 models (Gemma-2-2B-IT, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct) to keep total runtime realistic on free-tier Colab.

**Why it's novel**
This isn't "does the defense work" — it's "what does the internal geometry of safety failure look like." Layer-wise mapping + cross-method recovery + the decodable/elicitable/generated split together tell the field *where* the failure lives and *how general* it is, not just that it exists.

**Expected outcome**
A representation-geometry result showing whether unsafe capabilities cluster in specific layers, transfer across defense types, and sit close enough to "safe" activations to be re-elicited cheaply — a genuine RepE mechanism paper, strong for ICML/ICLR/NeurIPS main track and a strong fellowship work sample.
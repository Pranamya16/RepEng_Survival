# Data Card — Stage 1

## Sources
- Harmful base: CohereLabs/aya_redteaming (English subset) — 987 raw prompts
- Harmful volume: AdvBench (public CSV, llm-attacks repo) — 520 raw prompts
- Harmless: tatsu-lab/alpaca (no-input instructions, random sample matched to harmful count)
- Benign-but-adjacent: XSTest (public CSV, paul-rottger/xstest repo), safe-labeled subset — 250 prompts

## Processing
- Harmful prompts deduped (case-insensitive) across aya_redteaming + AdvBench: 1507 unique prompts.
- Harmless prompts sampled 1:1 to match harmful count, seed=42.
- Split with sklearn train_test_split, seed=42, same indices applied to harmful and harmless pools so pairs stay aligned.

## Splits (harmful + harmless combined, shuffled)
- probe_train.jsonl: 1204 rows (602 harmful + 602 harmless)
- probe_test.jsonl: 602 rows (301 harmful + 301 harmless)
- steering_construction.jsonl: 604 rows (302 harmful + 302 harmless)
- recovery_evaluation.jsonl: 604 rows (302 harmful + 302 harmless)
- benign_adjacent.jsonl: used only in Stage 6, never seen during probe/steering fitting

## Notes
- recovery_evaluation split must never be used during probe/steering construction (Stage 2-5) — held out for Stage 5 recovery testing only, per project_plan.md.
- Built locally via scripts/build_stage1_data.py (uv), not on Colab — this stage needs no GPU.

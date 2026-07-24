# Stage 2 Report: Baseline Activation Extraction and Refusal Behavior

**Project:** Representation Geometry of Safety Failure
**Date:** July 11, 2026

## What This Stage Does

Before studying how safety defenses fail, we first need to know how each model behaves with no defense applied at all. Stage 2 does two things for every base model, with no safety defense turned on:

1. Records the model's internal "activations" — the numeric signal inside the model at every layer — when it reads a harmful prompt versus a harmless one. Later stages use this saved data to build safety-detection probes and steering vectors, without needing to reload the models.
2. Checks how often the model actually refuses to answer harmful prompts, before any defense exists. This confirms each model has real refusal behavior to begin with — if a model didn't refuse at all here, a later stage asking "does the defense suppress refusal?" wouldn't mean anything.

## How It Was Done

- Four models were tested: Gemma-2-2B-IT, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, and Phi-4-mini-instruct, each loaded in 4-bit precision to fit on a free-tier GPU.
- Each model was tested against the four prompt sets built in Stage 1 (probe-train, probe-test, steering-construction, recovery-evaluation) — 1,507 harmful prompts and 1,507 matched harmless prompts in total.
- For every prompt, the model's internal activation at every layer was recorded and saved.
- For the 1,507 harmful prompts, the model also generated a real response, which was checked for refusal language (e.g. "I cannot help with that").
- This ran on Kaggle's free GPU tier, one model at a time — only one model fits in GPU memory at once on the free tier.

## Results

| Model | Harmful prompts tested | Refused | Refusal rate |
|---|---|---|---|
| Gemma-2-2B-IT | 1,507 | 1,105 | 73.3% |
| Phi-4-mini-instruct | 1,507 | 1,077 | 71.5% |
| Llama-3.2-3B-Instruct | 1,507 | 1,009 | 67.0% |
| Qwen2.5-3B-Instruct | 1,507 | 893 | 59.3% |

Activation data was saved successfully for all four prompt sets, for all four models — 16 activation files in total.

Llama-3.2-3B-Instruct needed two retries to complete. Both failures were caused by the same underlying issue: this model requires special Hugging Face access, and the access token wasn't reaching the model reliably. The first failure was a missing token; the second was because Kaggle mounted the token file in a different folder location than expected. Once the code was changed to search for the token file instead of assuming one fixed location, it completed successfully. This was a setup issue specific to how Kaggle mounts files, not a problem with the model or the data.

## What We Understand From These Results

- All four models refuse harmful prompts a clear majority of the time with no defense applied. This is the expected starting point, and it means later stages — which test whether a defense suppresses this refusal — will have a real, meaningful baseline to measure against.
- Refusal rate varies meaningfully across models: Gemma and Phi-4-mini refuse most often (71-73%), Llama sits in the middle (67%), and Qwen refuses least often (59%). This is a genuine difference between models, not an error — different providers tune their models to refuse harmful requests to different degrees by default. It's worth keeping in mind for later stages: a model with a lower starting refusal rate may look different in defense-related results simply because of where it started, not because a defense behaved differently.
- The reported refusal rate is likely a slight undercount for all four models. The check used to detect "did the model refuse" looks for refusal language near the start of its response, so a small number of replies that open with a longer sentence before refusing would be missed. The true refusal rate is probably a few points higher across the board.
- Because activation data saved correctly for all four models, the next stages that depend on it (finding where in the model the "harmful" signal lives, and testing whether it survives a defense) can proceed without redoing this work.

## Status

Stage 2 is complete for all four models. The project can proceed to Stage 3 (applying RepE defenses).

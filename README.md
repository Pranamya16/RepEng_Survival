# Representation Geometry of Safety Failure

Mapping how unsafe capabilities survive RepE safety defenses. See [Intro.md](Intro.md) for the pitch and [project_plan.md](project_plan.md) for the full stage-by-stage plan.

**All experiments run on Google Colab (free tier, T4 GPU).** Before running any notebook, read [claude.md](claude.md) — it covers Drive setup, the anti-disconnect cell, one-model-at-a-time GPU usage, and result checkpointing. These rules are mandatory for every notebook in `/notebooks`.

## Layout

```
/data          harmful/harmless prompt pairs, held-out test sets
/models        cached HF checkpoints, circuit-breaker weights (Drive only, not pushed)
/src           extraction.py, probes.py, steering.py, defenses.py, recovery.py, metrics.py
/notebooks     Colab-ready experiment notebooks, one per stage
/results       saved probe accuracies, steering magnitudes, plots
/paper         writeup, figures
```

## Models

Gemma-2-2B-IT, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Phi-4-mini-instruct — all loaded in 4-bit.

## Setup

```
pip install -q -U -r requirements.txt
```
(Torch is left as Colab's preinstalled, CUDA-matched version — don't reinstall it.)

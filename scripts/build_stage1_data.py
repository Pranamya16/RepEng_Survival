"""Stage 1 data construction — local (non-Colab) runner.

Builds harmful/harmless contrastive prompt pairs and the benign-but-adjacent
counterfactual set, then splits into probe_train / probe_test /
steering_construction / recovery_evaluation. Writes JSONL files + a data
card to /data.

This is the same logic as notebooks/01_data_construction.ipynb, minus the
Colab-only cells (Drive mount, anti-disconnect, HF login). AdvBench and
XSTest are pulled from their original public GitHub CSVs, since the HF
mirrors for both are gated. Run with:

    uv run --with datasets --with huggingface_hub --with pandas --with scikit-learn scripts/build_stage1_data.py

or, with a venv already set up:

    python scripts/build_stage1_data.py
"""

import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ADVBENCH_CSV = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
XSTEST_CSV = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"

SPLIT_RATIOS = {
    "probe_train": 0.4,
    "probe_test": 0.2,
    "steering_construction": 0.2,
    "recovery_evaluation": 0.2,
}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    random.seed(42)

    print("Loading harmful prompts...")
    aya = load_dataset("CohereLabs/aya_redteaming", split="english")
    aya_harmful = [row["prompt"] for row in aya]
    print(f"  aya_redteaming (English): {len(aya_harmful)} prompts")

    advbench_df = pd.read_csv(ADVBENCH_CSV)
    advbench_harmful = advbench_df["goal"].tolist()
    print(f"  AdvBench: {len(advbench_harmful)} prompts")

    harmful_raw = aya_harmful + advbench_harmful
    harmful_sources = ["aya_redteaming"] * len(aya_harmful) + ["advbench"] * len(advbench_harmful)

    print("Deduping harmful prompts...")
    seen = set()
    harmful_prompts, harmful_prompt_sources = [], []
    for prompt, source in zip(harmful_raw, harmful_sources):
        key = prompt.strip().lower()
        if key in seen or not key:
            continue
        seen.add(key)
        harmful_prompts.append(prompt.strip())
        harmful_prompt_sources.append(source)
    print(f"  Deduped harmful prompts: {len(harmful_prompts)}")

    print("Loading harmless counterparts (Alpaca)...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca_no_input = [row["instruction"] for row in alpaca if not row["input"]]
    harmless_prompts = random.sample(alpaca_no_input, len(harmful_prompts))
    harmless_prompt_sources = ["alpaca"] * len(harmless_prompts)
    print(f"  Harmless prompts (matched count): {len(harmless_prompts)}")

    print("Loading benign-but-adjacent set (XSTest, safe-labeled subset)...")
    xstest_df = pd.read_csv(XSTEST_CSV)
    benign_adjacent_prompts = xstest_df.loc[
        xstest_df["label"].str.strip().str.lower() == "safe", "prompt"
    ].tolist()
    print(f"  XSTest safe (benign-but-adjacent): {len(benign_adjacent_prompts)} prompts")

    print("Building splits...")
    n = len(harmful_prompts)
    indices = list(range(n))
    train_idx, temp_idx = train_test_split(indices, test_size=0.6, random_state=42)
    test_idx, temp_idx2 = train_test_split(temp_idx, test_size=2 / 3, random_state=42)
    steer_idx, recovery_idx = train_test_split(temp_idx2, test_size=0.5, random_state=42)

    splits = {
        "probe_train": train_idx,
        "probe_test": test_idx,
        "steering_construction": steer_idx,
        "recovery_evaluation": recovery_idx,
    }
    for name, idx in splits.items():
        print(f"  {name}: {len(idx)}")

    print("Writing splits and benign-adjacent set to /data...")
    for split_name, idx in splits.items():
        rows = []
        for i in idx:
            rows.append({"prompt": harmful_prompts[i], "label": "harmful", "source": harmful_prompt_sources[i]})
            rows.append({"prompt": harmless_prompts[i], "label": "harmless", "source": harmless_prompt_sources[i]})
        random.shuffle(rows)
        write_jsonl(DATA_DIR / f"{split_name}.jsonl", rows)
        print(f"  Wrote {split_name}.jsonl: {len(rows)} rows")

    benign_adjacent_rows = [{"prompt": p, "label": "benign_adjacent", "source": "xstest"} for p in benign_adjacent_prompts]
    write_jsonl(DATA_DIR / "benign_adjacent.jsonl", benign_adjacent_rows)
    print(f"  Wrote benign_adjacent.jsonl: {len(benign_adjacent_rows)} rows")

    print("Writing data card...")
    data_card = f"""# Data Card — Stage 1

## Sources
- Harmful base: CohereLabs/aya_redteaming (English subset) — {len(aya_harmful)} raw prompts
- Harmful volume: AdvBench (public CSV, llm-attacks repo) — {len(advbench_harmful)} raw prompts
- Harmless: tatsu-lab/alpaca (no-input instructions, random sample matched to harmful count)
- Benign-but-adjacent: XSTest (public CSV, paul-rottger/xstest repo), safe-labeled subset — {len(benign_adjacent_prompts)} prompts

## Processing
- Harmful prompts deduped (case-insensitive) across aya_redteaming + AdvBench: {len(harmful_prompts)} unique prompts.
- Harmless prompts sampled 1:1 to match harmful count, seed=42.
- Split with sklearn train_test_split, seed=42, same indices applied to harmful and harmless pools so pairs stay aligned.

## Splits (harmful + harmless combined, shuffled)
"""
    for split_name, idx in splits.items():
        data_card += f"- {split_name}.jsonl: {len(idx) * 2} rows ({len(idx)} harmful + {len(idx)} harmless)\n"
    data_card += "- benign_adjacent.jsonl: used only in Stage 6, never seen during probe/steering fitting\n"
    data_card += (
        "\n## Notes\n- recovery_evaluation split must never be used during probe/steering "
        "construction (Stage 2-5) — held out for Stage 5 recovery testing only, per project_plan.md.\n"
        "- Built locally via scripts/build_stage1_data.py (uv), not on Colab — this stage needs no GPU.\n"
    )
    (DATA_DIR / "DATA_CARD.md").write_text(data_card, encoding="utf-8")
    print(data_card)


if __name__ == "__main__":
    main()

"""Downloads Stage 2 Kaggle kernel outputs for all 4 models and builds one
consolidated baseline refusal-rate table + activation file inventory.

Run after all 4 stage2_* kernels have finished (COMPLETE or ERROR):
    .venv/Scripts/python.exe scripts/consolidate_stage2_results.py
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KAGGLE_EXE = BASE_DIR / ".venv" / "Scripts" / "kaggle.exe"
DOWNLOAD_DIR = BASE_DIR / "kaggle_downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

KERNELS = {
    "gemma-2-2b-it": "pranamyadeshpande/repeng-survival-stage-2-gemma-2-2b-it",
    "llama-3.2-3b-instruct": "pranamyadeshpande/repeng-survival-stage-2-llama-3-2-3b-instruct",
    "qwen2.5-3b-instruct": "pranamyadeshpande/repeng-survival-stage-2-qwen2-5-3b-instruct",
    "phi-4-mini-instruct": "pranamyadeshpande/repeng-survival-stage-2-phi-4-mini-instruct",
}

SPLIT_NAMES = ["probe_train", "probe_test", "steering_construction", "recovery_evaluation"]


def download_output(short_name, ref):
    out_dir = DOWNLOAD_DIR / short_name
    out_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        [str(KAGGLE_EXE), "kernels", "output", ref, "-p", str(out_dir)],
        capture_output=True, text=True,
    )
    return out_dir, result.returncode == 0, result.stdout + result.stderr


def main():
    rows = []
    errors = []

    for short_name, ref in KERNELS.items():
        print(f"=== {short_name} ({ref}) ===")
        out_dir, ok, log = download_output(short_name, ref)
        if not ok:
            print(f"  Download failed: {log[-500:]}")
            errors.append({"model": short_name, "issue": "download_failed", "detail": log[-1000:]})
            continue

        results_dir = out_dir / "results"
        if not results_dir.exists():
            # kaggle output sometimes nests under the kernel slug directory
            candidates = list(out_dir.glob("**/results"))
            results_dir = candidates[0] if candidates else None

        if results_dir is None or not results_dir.exists():
            print(f"  No results/ folder found in downloaded output.")
            errors.append({"model": short_name, "issue": "no_results_dir"})
            continue

        err_file = results_dir / f"stage2_error_{short_name}.json"
        if err_file.exists():
            with open(err_file) as f:
                err_data = json.load(f)
            print(f"  Model-level error recorded: {err_data.get('error')}")
            errors.append({"model": short_name, "issue": "model_error", "detail": err_data.get("error")})

        for split_name in SPLIT_NAMES:
            refusal_file = results_dir / f"stage2_refusal_{short_name}_{split_name}.json"
            act_file = results_dir / "activations" / f"{short_name}_{split_name}.pt"
            if refusal_file.exists():
                with open(refusal_file) as f:
                    r = json.load(f)
                rows.append({
                    "model": short_name,
                    "split": split_name,
                    "n_harmful": r.get("n_harmful"),
                    "refusal_rate": r.get("refusal_rate"),
                    "activations_saved": act_file.exists(),
                })
            else:
                rows.append({
                    "model": short_name,
                    "split": split_name,
                    "n_harmful": None,
                    "refusal_rate": None,
                    "activations_saved": act_file.exists(),
                })

    print("\n=== Baseline refusal rate table ===")
    header = f"{'model':<24} {'split':<24} {'n_harmful':>10} {'refusal_rate':>14} {'activations':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        rr = f"{row['refusal_rate']:.3f}" if row["refusal_rate"] is not None else "N/A"
        nh = row["n_harmful"] if row["n_harmful"] is not None else "N/A"
        print(f"{row['model']:<24} {row['split']:<24} {str(nh):>10} {rr:>14} {str(row['activations_saved']):>12}")

    if errors:
        print("\n=== Errors ===")
        for e in errors:
            print(f"- {e['model']}: {e['issue']} {e.get('detail', '')[:200]}")

    summary_path = DOWNLOAD_DIR / "stage2_consolidated_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"rows": rows, "errors": errors}, f, indent=2)
    print(f"\nSaved consolidated summary to {summary_path}")


if __name__ == "__main__":
    main()

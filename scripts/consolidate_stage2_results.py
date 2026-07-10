"""Downloads Stage 2 Kaggle kernel outputs for all 4 models and builds one
consolidated baseline refusal-rate table + activation file inventory.

Gemma runs in its own kernel; Llama/Qwen/Phi-4-mini run together in one
combined kernel (Kaggle rejects a second concurrent batch GPU push, so
they were merged into a single sequential run instead of 3 separate ones).

Run after both kernels have finished (COMPLETE or ERROR):
    .venv/Scripts/python.exe scripts/consolidate_stage2_results.py
"""

import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KAGGLE_EXE = BASE_DIR / ".venv" / "Scripts" / "kaggle.exe"
DOWNLOAD_DIR = BASE_DIR / "kaggle_downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

KERNEL_REFS = {
    "gemma": "pranamyadeshpande/repeng-survival-stage-2-gemma-2-2b-it",
    "remaining3": "pranamyadeshpande/repeng-survival-stage-2-remaining-models-v2",
}
MODELS_BY_KERNEL = {
    "gemma": ["gemma-2-2b-it"],
    "remaining3": ["llama-3.2-3b-instruct", "qwen2.5-3b-instruct", "phi-4-mini-instruct"],
}

SPLIT_NAMES = ["probe_train", "probe_test", "steering_construction", "recovery_evaluation"]


def download_output(kernel_key, ref):
    out_dir = DOWNLOAD_DIR / kernel_key
    out_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        [str(KAGGLE_EXE), "kernels", "output", ref, "-p", str(out_dir), "--force"],
        capture_output=True, text=True,
    )
    return out_dir, result.returncode == 0, result.stdout + result.stderr


def find_results_dir(out_dir):
    results_dir = out_dir / "results"
    if results_dir.exists():
        return results_dir
    candidates = list(out_dir.glob("**/results"))
    return candidates[0] if candidates else None


def main():
    rows = []
    errors = []

    for kernel_key, ref in KERNEL_REFS.items():
        print(f"=== downloading {kernel_key} ({ref}) ===")
        out_dir, ok, log = download_output(kernel_key, ref)
        if not ok:
            print(f"  Download failed: {log[-500:]}")
            for model in MODELS_BY_KERNEL[kernel_key]:
                errors.append({"model": model, "issue": "download_failed", "detail": log[-1000:]})
            continue

        results_dir = find_results_dir(out_dir)
        if results_dir is None:
            print("  No results/ folder found in downloaded output.")
            for model in MODELS_BY_KERNEL[kernel_key]:
                errors.append({"model": model, "issue": "no_results_dir"})
            continue

        for short_name in MODELS_BY_KERNEL[kernel_key]:
            err_file = results_dir / f"stage2_error_{short_name}.json"
            if err_file.exists():
                with open(err_file) as f:
                    err_data = json.load(f)
                print(f"  {short_name}: model-level error recorded: {err_data.get('error')}")
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

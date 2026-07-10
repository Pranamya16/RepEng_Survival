"""Stage 2 (Kaggle runner, single model) — Phi-4-mini-instruct baseline
activation extraction & refusal behavior. Account: deshpandepranamya.

One of 4 parallel single-model kernels (see kaggle/stage2_*) run across 4
separate Kaggle accounts so all 4 models process simultaneously instead of
sequentially. Same logic as kaggle/stage2_baseline_extraction, just scoped
to one model. Not gated on HF, so no token needed.
"""

import os
import sys
import json
import gc
import subprocess
import traceback

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes"],
    check=True,
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_URL = "https://github.com/Pranamya16/RepEng_Survival.git"
REPO_DIR = "/kaggle/working/RepEng_Survival"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

sys.path.append(os.path.join(REPO_DIR, "src"))
from extraction import extract_activations, save_activations
from metrics import refusal_classifier

DATA_DIR = os.path.join(REPO_DIR, "data")
RESULTS_DIR = "/kaggle/working/results"
ACTIVATIONS_DIR = os.path.join(RESULTS_DIR, "activations")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ACTIVATIONS_DIR, exist_ok=True)


def save_result(name, data, directory=RESULTS_DIR):
    with open(os.path.join(directory, name), "w") as f:
        json.dump(data, f, indent=2)


def load_result_if_exists(name, directory=RESULTS_DIR):
    path = os.path.join(directory, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


SPLIT_NAMES = ["probe_train", "probe_test", "steering_construction", "recovery_evaluation"]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


splits = {name: load_jsonl(os.path.join(DATA_DIR, f"{name}.jsonl")) for name in SPLIT_NAMES}
for name, rows in splits.items():
    print(f"{name}: {len(rows)} rows")

MODELS = {
    "phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

MAX_NEW_TOKENS = 40

for short_name, hf_id in MODELS.items():
    print(f"=== {short_name} ===")
    model = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=bnb_config, device_map="auto")

        for split_name, rows in splits.items():
            act_path = os.path.join(ACTIVATIONS_DIR, f"{short_name}_{split_name}.pt")
            refusal_result_name = f"stage2_refusal_{short_name}_{split_name}.json"

            if os.path.exists(act_path) and load_result_if_exists(refusal_result_name):
                print(f"  Skipping {split_name}, already done.")
                continue

            print(f"  Extracting activations: {split_name} ({len(rows)} prompts)")
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
                )
                for row in rows
            ]
            acts = extract_activations(model, tokenizer, formatted)
            payload = {
                "activations": acts,
                "labels": [row["label"] for row in rows],
                "sources": [row["source"] for row in rows],
                "prompts": [row["prompt"] for row in rows],
            }
            save_activations(payload, act_path)

            print(f"  Checking baseline refusal on harmful prompts in {split_name}")
            harmful_rows = [row for row in rows if row["label"] == "harmful"]
            refused = 0
            samples = []
            for row in harmful_rows:
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True,
                ).to(model.device)
                output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                is_refusal = refusal_classifier(generated)
                refused += int(is_refusal)
                if len(samples) < 5:
                    samples.append({"prompt": row["prompt"], "generated": generated, "refused": is_refusal})

            refusal_result = {
                "model": short_name,
                "split": split_name,
                "n_harmful": len(harmful_rows),
                "n_refused": refused,
                "refusal_rate": refused / len(harmful_rows) if harmful_rows else None,
                "samples": samples,
            }
            save_result(refusal_result_name, refusal_result)
            print(f"    refusal_rate = {refusal_result['refusal_rate']}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"FAILED on {short_name}: {e!r}")
        print(tb)
        save_result(f"stage2_error_{short_name}.json", {"model": short_name, "error": str(e) or repr(e), "traceback": tb})

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

print("Stage 2 (Kaggle, phi-4-mini-instruct) run complete.")

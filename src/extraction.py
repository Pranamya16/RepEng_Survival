"""Activation extraction hooks: pull residual-stream activations per layer for a given model/prompt set."""

import torch


@torch.no_grad()
def extract_activations(model, tokenizer, prompts):
    """Last-token residual-stream activation at every layer, for each prompt.

    Processes one prompt at a time (no padding) to keep last-token indexing
    unambiguous regardless of a model's default padding side.

    Returns a tensor [num_prompts, num_layers+1, hidden_dim] — layer 0 is the
    embedding output, layers 1..N are each transformer block's output
    (matches `output_hidden_states` from a HF forward pass). Stored as
    float16 to keep saved activation files a manageable size.
    """
    model.eval()
    all_acts = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model(**encoded, output_hidden_states=True)
        layer_acts = torch.stack([h[0, -1].half().cpu() for h in outputs.hidden_states], dim=0)
        all_acts.append(layer_acts)
    return torch.stack(all_acts, dim=0)


def save_activations(activations, path):
    torch.save(activations, path)


def load_activations(path):
    return torch.load(path)

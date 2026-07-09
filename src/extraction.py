"""Activation extraction hooks: pull residual-stream activations per layer for a given model/prompt set."""


def extract_activations(model, tokenizer, prompts, layers=None):
    raise NotImplementedError


def save_activations(activations, path):
    raise NotImplementedError


def load_activations(path):
    raise NotImplementedError

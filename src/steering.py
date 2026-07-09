"""Steering vector construction (diff-of-means, CAA-style) and application at inference time."""


def build_steering_vector(harmful_activations, harmless_activations):
    raise NotImplementedError


def apply_steering(model, vector, layer, alpha):
    raise NotImplementedError


def remove_steering(model):
    raise NotImplementedError

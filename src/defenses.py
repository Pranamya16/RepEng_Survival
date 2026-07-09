"""Apply RepE defenses: refusal-direction ablation (Defense A) and circuit breakers (Defense B)."""


def build_refusal_direction(harmful_activations, harmless_activations):
    raise NotImplementedError


def apply_refusal_ablation(model, direction):
    raise NotImplementedError


def load_circuit_breaker(model_id):
    raise NotImplementedError


def apply_circuit_breaker(model, checkpoint_path):
    raise NotImplementedError

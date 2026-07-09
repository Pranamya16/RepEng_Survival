"""Recovery strength, decodability score, elicitation score, generation-quality checks."""


def decodability_score(probe_accuracy):
    raise NotImplementedError


def elicitation_score(recovery_success_rate, min_alpha):
    raise NotImplementedError


def fluency_check(generated_text):
    raise NotImplementedError


def refusal_classifier(generated_text):
    raise NotImplementedError

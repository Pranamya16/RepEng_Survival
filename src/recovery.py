"""Re-elicitation attacks: same-method and cross-method recovery of suppressed behavior."""


def same_method_recovery(defended_model, steering_vector, alphas):
    raise NotImplementedError


def cross_method_recovery(defended_model, other_method_vector, alphas):
    raise NotImplementedError


def min_alpha_for_flip(defended_model, steering_vector, alphas, classifier):
    raise NotImplementedError

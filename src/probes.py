"""Linear probe training/eval, layer-wise sweep for harmful vs. harmless classification."""


def train_probe(X_train, y_train):
    raise NotImplementedError


def eval_probe(probe, X_test, y_test):
    raise NotImplementedError


def layer_sweep(activations_by_layer, labels):
    raise NotImplementedError

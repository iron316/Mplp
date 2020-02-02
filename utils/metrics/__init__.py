from .evaluation import classification_eval, regression_eval, binary_eval
EVAL = {"binary": binary_eval,
        "multiclass": classification_eval,
        "regression": regression_eval}

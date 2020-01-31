from .binary import BinaryModel
from .multiclass import MulticlassModel
from .regression import RegressionModel

MODELS = {"binary": BinaryModel,
          "multiclass": MulticlassModel,
          "regression": RegressionModel
          }

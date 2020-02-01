from .binary import BinaryModel
from .multiclass import MulticlassModel
from .regression import RegressionModel
from .base import BaseModule

MODELS = {"binary": BinaryModel,
          "multiclass": MulticlassModel,
          "regression": RegressionModel
          }

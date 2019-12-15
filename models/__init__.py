from models.binary import BinaryModel
from models.multiclass import MulticlassModel
from models.regression import RegressionModel

MODELS = {"binary": BinaryModel,
          "multiclass": MulticlassModel,
          "regression": RegressionModel
          }

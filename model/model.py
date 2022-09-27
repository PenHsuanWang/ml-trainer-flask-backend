from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class Model(ABC):

    def __init__(self):
        self._model = None

    def _init_model(self):
        raise NotImplementedError

    def fit(self, x, y):
        print("going to train model")
        self._model.fit(x, y)

    def predict_proba(self, x) -> list:
        pred_proba_result = self._model.predict_proba(x)
        target_pred_proba = pred_proba_result[:, 1]

        return target_pred_proba

    def predict(self, x, pred_proba_cut_point=0.5) -> list:
        target_pred_proba = self.predict_proba(x)
        pred_casting_to_binary = list(map(lambda x: 0 if x < pred_proba_cut_point else 1, target_pred_proba))

        return pred_casting_to_binary



class ModelSklearnRFClassifier(Model, ABC):

    def __init__(self, *args, **kwargs):
        super(ModelSklearnRFClassifier).__init__()
        self._init_model(*args, **kwargs)

    def _init_model(self, *args, **kwargs):

        self._model = RandomForestClassifier(*args, **kwargs)


    def __str__(self):
        return "ModelSklearnRFClassifier"



class ModelXGBClassifier(Model, ABC):

    def __init__(self, *args, **kwargs):
        super(ModelXGBClassifier).__init__()
        self._init_model(*args, **kwargs)

    def _init_model(self, *args, **kwargs):

        self._model = XGBClassifier(
            *args, **kwargs
        )


    def __str__(self):
        return "ModelXGBClassifier"

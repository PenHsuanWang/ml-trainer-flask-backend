from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier


class Model(ABC):

    def __init__(self):
        self._model = None

    def _init_model(self):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x, predict_proba_cut_point) -> list:
        raise NotImplementedError

    def predict_proba(self, x) -> list:
        raise NotImplementedError


class SklearnRandomForestClassifier(Model, ABC):

    def __init__(self, *args, **kwargs):
        super(SklearnRandomForestClassifier).__init__()
        self._init_model(*args, **kwargs)

    def _init_model(self, *args, **kwargs):

        self._model = RandomForestClassifier(*args, **kwargs)

    def fit(self, x, y):
        print("going to train model")
        self._model.fit(x, y)

    def predict(self, x, pred_proba_cut_point=0.5) -> list:

        # pred_proba_result = self._model.predict_proba(x)
        # target_pred_proba = pred_proba_result[:, 1]

        target_pred_proba = self.predict_proba(x)
        pred_casting_to_binary = list(map(lambda x:0 if x < pred_proba_cut_point else 1, target_pred_proba))

        return pred_casting_to_binary

    def predict_proba(self, x) -> list:

        pred_proba_result = self._model.predict_proba(x)
        target_pred_proba = pred_proba_result[:, 1]

        return target_pred_proba


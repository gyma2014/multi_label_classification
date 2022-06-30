

from abc import abstractmethod

class MultiLabelModel(object):

    @abstractmethod
    def fit_impl(self, data, **kargs):
        raise NotImplementedError("fit_impl is not implemented, model is not train-able")

    @abstractmethod
    def predict_impl(self, X, **kwargs):
        raise NotImplementedError("predict_impl is not implemented, model cannot repdict")

    def fit(self, **kwargs):
        return self.fit_impl(**kwargs)

    def predict(self, X):
        return self.predict_impl(X)

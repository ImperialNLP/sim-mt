from ignite import metrics as ig_metrics

from ..metrics import Metric


class Loss:
    """Accumulates and computes correctly training and validation losses."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._loss = 0
        self._denom = 0
        self.batch_loss = 0

    def update(self, loss, n_items):
        # Store last batch loss
        self.batch_loss = loss.item()
        # Add it to cumulative loss
        self._loss += self.batch_loss
        # Normalize batch loss w.r.t n_items
        self.batch_loss /= n_items
        # Accumulate n_items inside the denominator
        self._denom += n_items

    def get(self):
        if self._denom == 0:
            return 0
        return self._loss / self._denom

    @property
    def denom(self):
        return self._denom


class Precision:
    """Wrapper metric around `pytorch-ignite`."""
    def __init__(self, is_multilabel=True):
        self.is_multilabel = is_multilabel
        self.__metric = ig_metrics.Precision(
            average=True, is_multilabel=self.is_multilabel)

    def update(self, y_pred, y):
        """Tensors should have N x n_labels dimensions with N = batch_size."""
        self.__metric.update((y_pred, y))

    def compute(self):
        """Once the updates are over, this returns the actual metric."""
        val = 100 * self.__metric.compute()
        return Metric('PRECISION', val, higher_better=True)


class Recall:
    """Wrapper metric around `pytorch-ignite`."""
    def __init__(self, is_multilabel=True):
        self.is_multilabel = is_multilabel
        self.__metric = ig_metrics.Recall(
            average=True, is_multilabel=self.is_multilabel)

    def update(self, y_pred, y):
        """Tensors should have N x n_labels dimensions with N = batch_size."""
        self.__metric.update((y_pred, y))

    def compute(self):
        """Once the updates are over, this returns the actual metric."""
        val = 100 * self.__metric.compute()
        return Metric('RECALL', val, higher_better=True)


class F1:
    """Wrapper metric around `pytorch-ignite`."""
    def __init__(self, is_multilabel=True):
        self.is_multilabel = is_multilabel
        # Create underlying metrics
        self.__precision = ig_metrics.Precision(
            average=False, is_multilabel=self.is_multilabel)

        self.__recall = ig_metrics.Recall(
            average=False, is_multilabel=self.is_multilabel)

        num = self.__precision * self.__recall * 2
        denom = self.__precision + self.__recall + 1e-20
        f1 = num / denom
        self.__metric = ig_metrics.MetricsLambda(
            lambda t: t.mean().item(), f1)

    def update(self, y_pred, y):
        """Tensors should have N x n_labels dimensions with N = batch_size."""
        self.__precision.update((y_pred, y))
        self.__recall.update((y_pred, y))

    def compute(self):
        val = 100 * self.__metric.compute()
        return Metric('F1', val, higher_better=True)

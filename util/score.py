import typing as T
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mlnext import check_ndim
from mlnext import check_shape
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score as roc_auc_score_
from sklearn.metrics._ranking import _binary_clf_curve

__all__ = [
    'recall_segments',
    'auc_recall_segments',
    'PRCurve',
    'ConfusionMatrix',
    'pr_curve',
    'roc_auc_score',
    'nanargmin',
    'nanargmax'
]


def nanargmin(a: np.ndarray, axis: int = None, default: int = 0) -> int:
    """np.nanargmin but instead of raising an exception for all nan-slices
    it returns ``default``.

    Args:
        a (np.ndarray): Array.
        axis (int, optional): Axis. Defaults to None.
        default (int, optional): Value to return for all nan-slices.
          Defaults to -1.

    Returns:
        int: Returns the index of the argument that is minimal.
    """
    try:
        return np.nanargmin(a, axis)
    except ValueError:
        return default


def nanargmax(a: np.ndarray, axis: int = None, default: int = 0) -> int:
    """np.nanargmax but instead of raising an exception for all nan-slices
    it returns ``default``.

    Args:
        a (np.ndarray): Array.
        axis (int, optional): Axis. Defaults to None.
        default (int, optional): Value to return for all nan-slices.
          Defaults to -1.

    Returns:
        int: Returns the index of the argument that is maximal.
    """
    try:
        return np.nanargmax(a, axis)
    except ValueError:
        return default

def _recall_segments_curve(
    y_score: np.ndarray,
    y: np.ndarray,
    *,
    thresholds: T.List[float],
    k: float = 0
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Calculates ``recall_segments`` for various ``threshold``.

    Args:
        y_score (np.ndarray): Score.
        y (np.ndarray): Ground truth labels.
        thresholds (np.ndarray): List of thresholds.

    Returns:
        T.Tuple[np.ndarray, np.ndarray]: Returns the detected and total
          segments.
    """

    y_score, y = np.array(y_score).squeeze(), np.array(y).squeeze()
    check_ndim(y_score, y, ndim=1), check_shape(y_score, y)

    # find start and end of anomaly segments
    y = pd.Series(y)
    # true for index i when y[i - 1] = 0 and y[i] = 1
    start = (y > y.shift(1, fill_value=0))
    # true for index i when y[i] = 1 and y[i + 1] = 0
    end = (y > y.shift(-1, fill_value=0))
    # get indices where true
    start, end = np.flatnonzero(start), np.flatnonzero(end)

    # count segments where at least k% are detected
    detected = np.zeros(len(thresholds))
    total = np.ones(len(thresholds)) * len(start)
    for i, threshold in enumerate(thresholds):
        y_hat = np.where(y_score >= threshold, 1, 0)
        detected[i] = np.sum([
            np.sum(y_hat[s:(e + 1)]) > ((k / 100) * (e + 1 - s))
            for s, e in zip(start, end)
        ])

    return detected, total


def _recall_segments(
    y_hat: np.ndarray,
    y: np.ndarray,
    *,
    k: float = 0
) -> T.Tuple[int, int]:
    """Calculates the percentage of anomaly segments that are correctly
    detected. The parameter ``k`` [in %] controls how much of a segments needs
    to be detected for it being counted as detected.

    Args:
        y_hat (np.ndarray): Label predictions.
        y (np.ndarray): Ground Truth.
        k (float): Percentage ([0, 100]) of points in a segment for it to
          count. For K = 0, then at least one point has to be detected.
          For K = 100, then every point in the segment has to be correctly
          detected. Default: 0.

    Returns:
        T.Tuple[int, int]: Returns the number of detected and total segments.
    """
    y_hat, y = np.array(y_hat).squeeze(), np.array(y).squeeze()
    check_ndim(y_hat, y, ndim=1), check_shape(y_hat, y)

    # find start and end of anomaly segments
    y = pd.Series(y)
    # true for index i when y[i - 1] = 0 and y[i] = 1
    start = (y > y.shift(1, fill_value=0))
    # true for index i when y[i] = 1 and y[i + 1] = 0
    end = (y > y.shift(-1, fill_value=0))
    # get indices where true
    start, end = np.flatnonzero(start), np.flatnonzero(end)

    # count segments where at least k% are detected
    detected = np.sum([
        np.sum(y_hat[s:(e + 1)]) > ((k / 100) * (e + 1 - s))
        for s, e in zip(start, end)
    ])

    return detected, len(start)


def recall_segments(
    y_hat: np.ndarray,
    y: np.ndarray,
    *,
    k: float = 0
) -> float:
    """Calculates the percentage of anomaly segments that are correctly
    detected. The parameter ``k`` [in %] controls how much of a segments needs
    to be detected for it being counted as detected.

    Args:
        y_hat (np.ndarray): Label predictions.
        y (np.ndarray): Ground Truth.
        k (float): Percentage ([0, 100]) of points in a segment for it to
          count. For K = 0, then at least one point has to be detected.
          For K = 100, then every point in the segment has to be correctly
          detected. Default: 0.

    Returns:
        float: Returns the percentage of anomaly segments that have detected.
    """
    detected, total = _recall_segments(y_hat, y, k=k)
    return detected / total


def auc_recall_segments(
    y_hat: np.ndarray,
    y: np.ndarray,
    *,
    interval: int = 5
) -> float:
    """Calculates the area under the recall_segments curve. Thereby, the
    ``recall_segments`` is calculates for various k from [0, 100] with the
    step size ``interval``.

    Args:
        y_hat (np.ndarray): Label predictions.
        y (np.ndarray): Ground truth.
        interval (float, optional): Step size for k in [0, 100].
          Defaults to 5.

    Returns:
        float: Returns the AUC of recall_segments.
    """
    K = list(range(0, 101, interval))
    K[-1] = 99.9999  # type:ignore
    recalls = [recall_segments(y_hat, y, k=k) for k in K]
    return auc(np.r_[K, 100], np.r_[recalls, 0]) / 100


@dataclass
class ConfusionMatrix:
    """`ConfusionMatrix` is a confusion matrix for a binary classification
    problem. See https://en.wikipedia.org/wiki/Confusion_matrix.

    Args:
      TP (int): true positives, the number of samples from the positive class
        that are correctly assigned to the positive class.
      FN (int): false negatives, the number of samples from the positive
        class that are wrongly assigned to the negative class.
      TN (int): true negatives, the number of samples from the negative class
        that are correctly assigned to negative class.
      FP (int): true negatives, the number of samples from the negative class
        that are wrongly assigned to the positive class.
    """
    TP: int = 0  # True Positives
    FN: int = 0  # False Negatives
    TN: int = 0  # True Negative
    FP: int = 0  # False Positive
    DS: int = 0  # detected anomaly segments
    TS: int = 0  # total number of anomaly segments

    def __add__(self, cm: 'ConfusionMatrix') -> 'ConfusionMatrix':
        """Overrides the add operator.

        Returns:
            ConfusionMatrix: Returns a new matrix with feature-wise added
            values.
        """
        return ConfusionMatrix(
            TP=self.TP + cm.TP,
            FN=self.FN + cm.FN,
            TN=self.TN + cm.TN,
            FP=self.FP + cm.FP,
            DS=self.DS + cm.DS,
            TS=self.TS + cm.TS
        )

    def __str__(self) -> str:
        """Creates a string representation of the matrix.

        Returns:
            str: Returns a string representation of the confusion matrix.
        """
        rows = [
            '{:<3s} {:^6s} {:^6s}'.format('P\\A', '1', '0'),
            '{:<3s} {:^6.0f} {:^6.0f}'.format('1', self.TP, self.FP),
            '{:<3s} {:^6.0f} {:^6.0f}'.format('0', self.FN, self.TN),
            *[f'{k}: {v:.4f}' for k, v in self.metrics().items()]
        ]

        return '\n'.join(rows)

    @property
    def accuracy(self) -> float:
        """Calculates the accuracy `(TP + TN) / (TP + TN + FP + FN)`.

        Returns:
            np.ndarray: Returns the accuracy.
        """
        return ((self.TP + self.TN) /
                (self.TP + self.TN + self.FN + self.FP))

    @property
    def precision(self) -> float:
        """Calculates the precision `TP / (TP + FP)`.

        Returns:
            float: Returns the precision.
        """
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        """Calculates the recall `TP / (TP + FN)`.

        Returns:
            float: Returns the recall.
        """
        return self.TP / (self.TP + self.FN)

    @property
    def f1(self) -> float:
        """Calculates the F1-Score
        `2 * (precision * recall) / (precision + recall)`.

        Returns:
            np.ndarray: Returns the F1-score.
        """
        return ((2 * self.precision * self.recall) /
                (self.precision + self.recall))

    @property
    def recall_segments(self) -> float:
        """Calculates the percentage of detected anomaly segments.

        Returns:
            float: Returns the percentage of detected segments.
        """
        return self.DS / self.TS

    def metrics(self) -> T.Dict[str, float]:
        """Returns all metrics.

        Returns:
            T.Dict[str, float]: Returns an mapping of all performance metrics.
        """
        return {
            'accuracy': self.accuracy,
            'f1': self.f1,
            'recall': self.recall,
            'precision': self.precision,
            'recall_segments': self.recall_segments
        }


@dataclass
class PRCurve:
    """Container for the result of `pr_curve`. Additionally computes the
    F1-score for each threshold. Can be indexed and returns a
    `ConfusionMatrix` for the i-th threshold.

    Args:
        tps (np.ndarray): An increasing count of true positives, at index i
          being the number of positive samples assigned a score >=
          thresholds[i].
        fns (np.ndarray): A count of false negatives, at index i being the
          number of positive samples assigned a score < thresholds[i].
        tns (np.ndarray): A count of true negatives, at index i being the
          number of negative samples assigned a score < thresholds[i].
        fps (np.ndarray): A count of false positives, at index i being the
          number of negative samples assigned a score >= thresholds[i].
        dss (np.ndarray): A count of detected anomaly segments, at index i
          being the number of segments where at least 1 point has a score
          >= thresholds[i].
        tss (np.ndarray): The total number of anomaly segments. The same
          for every threshold.
        precision (np.ndarray): Precision values such that element i is the
          precision of predictions with score >= thresholds[i].
        recall (np.ndarray): Decreasing recall values such that element i is
          the recall of predictions with score >= thresholds[i].
        thresholds (np.ndarray): Increasing thresholds on the decision function
          used to compute precision and recall.
        f1 (np.ndarray): F1 score values such that element i is the f1 score
          of predictions with score >= thresholds[i].
    """

    tps: np.ndarray  # true positives
    fns: np.ndarray  # false negatives
    tns: np.ndarray  # true negatives
    fps: np.ndarray  # false positives
    dss: np.ndarray  # number of detected anomaly segments
    tss: np.ndarray  # total number of anomaly segments

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray

    def __str__(self) -> str:
        """Creates a string representation of the curve.

        Returns:
            str: Returns a string representation of the pr-curve.
        """

        header = (' {:^6s} ' * 12).format(
            'TH', 'ACC', 'F1', 'PRC', 'RCL', 'RS', 'TP', 'FN', 'TN', 'FP',
            'DS', 'TS'
        )

        fmt = ' {:^6.4f} ' * 6 + ' {:^6.0f} ' * 6
        rows = [
            fmt.format(*row)
            for row in
            zip(self.thresholds, self.accuracy, self.f1, self.precision,
                self.recall, self.recall_segments, self.tps, self.fns,
                self.tns, self.fps, self.dss, self.tss
                )]
        return '\n'.join([header, *rows, f'AUC: {self.auc:.4f}'])

    def __getitem__(self, i: int) -> ConfusionMatrix:
        """Creates a confusion matrix for threshold at index i.

        Args:
            idx (int): Threshold index.

        Raises:
            IndexError: Raised if the index is invalid.

        Returns:
            ConfusionMatrix: Returns the confusion matrix for threshold at
            index i.
        """

        if i < 0 or i >= len(self.thresholds):
            raise IndexError(f'Index {i} out of range.')

        return ConfusionMatrix(
            TP=self.tps[i],
            FN=self.fns[i],
            TN=self.tns[i],
            FP=self.fps[i],
            DS=self.dss[i],
            TS=self.tss[i]
        )

    def __len__(self) -> int:
        """Returns the number of thresholds that make up the curve.

        Returns:
            int: Returns the number of thresholds.
        """
        return len(self.thresholds)

    def __iter__(self) -> T.Iterator[ConfusionMatrix]:
        """Creates an iterator over the curve.

        Yields:
            T.Iterator[ConfusionMatrix]: Returns an iterator over the pr curve.
            At index i, the iterator returns a ConfusionMatrix for the i-th
            threshold.
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def accuracy(self) -> np.ndarray:
        """Calculates the accuracy where at index i, the accuracy is the
        percentage of correctly assigned samples.

        Returns:
            np.ndarray: Returns the accuracy.
        """
        return ((self.tps + self.tns) /
                (self.tps + self.tns + self.fns + self.fps))

    @property
    def auc(self) -> float:
        """Calculates the area-under-curve (auc).

        Returns:
            float: Returns the area-under-curve (auc) for the precision-recall
            curve.
        """
        # insert (0,1) such that the curve starts from 0
        return auc(np.r_[self.recall, 0], np.r_[self.precision, 1])

    @property
    def f1(self) -> np.ndarray:
        """Calculates the F1-score.

        Returns:
            np.ndarray: Returns the F1-score.
        """
        return ((2 * self.precision * self.recall) /
                (self.precision + self.recall))

    @property
    def recall_segments(self) -> np.ndarray:
        """Calculates the percentage of detected anomaly segments.

        Returns:
            np.ndarray: Returns the segment recall.
        """
        return self.dss / self.tss

    def to_tensorboard(self) -> T.Dict[str, T.Any]:
        """Converts the container to keyword arguments for Tensorboard.
            See https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md.

            Returns:
                T.Dict[str, T.Any]: Returns the pr-curve format expected for
                Tensorboard.
            """  # noqa
        return {
            'true_positive_counts': self.tps,
            'false_positive_counts': self.fps,
            'true_negative_counts': self.tns,
            'false_negative_counts': self.fns,
            'precision': self.precision,
            'recall': self.recall,
            'num_thresholds': len(self.thresholds)
        }


def pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    pos_label: T.Union[str, int] = None,
    sample_weight: T.Union[T.List, np.ndarray] = None
) -> PRCurve:
    """Computes precision-recall pairs for different probability thresholds for
    binary classification tasks.

    Adapted from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html.
    Changed the return value to PRCurve which encapsulates not only the
    recall, precision and thresholds but also the tps, fps, tns and fns. Thus,
    we can obtain all necessary parameters that are required for the logging of
    a pr-curve in tensorboard (https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md).
    Furthermore, we can you use results for further processing.

    Args:
        y_true (np.ndarray): Positive labels either {-1, 1} or {0, 1}.
          Otherwise, pos_label needs to be given.
        y_score (np.ndarray):  Target scores in range [0, 1].
        pos_label (int, optional):The label of the positive class.
          When pos_label=None, if y_true is in {-1, 1} or {0, 1},
          pos_label is set to 1, otherwise an error will be raised.
          Defaults to None.
        sample_weight (T.Union[T.List, np.ndarray], optional): Sample weights.
          Defaults to None.

    Returns:
        PRCurve: Returns a PRCurve container for the results.
    """  # noqa

    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    fns = tps[-1] - tps
    tns = fps[-1] - fps

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    dss, tss = _recall_segments_curve(y_score, y_true, thresholds=thresholds)

    return PRCurve(
        tps[sl], fns[sl], tns[sl], fps[sl], dss[sl], tss[sl],
        precision[sl], recall[sl], thresholds[sl]
    )


def roc_auc_score(y: np.ndarray, y_score: np.ndarray) -> float:
    """Wrapper around ``sklearn.metrics.roc_auc_score`` which does not raise
    an exception for the ill-defined cases.

    Args:
        y (np.ndarray): Ground truth.
        y_score (np.ndarray): Prediction scores.

    Returns:
        float: Returns the auc roc score.
    """
    try:
        return roc_auc_score_(y, y_score)
    except:  # noqa
        return 0.0

import numpy as np
import typing as T
from .score import *
import dataclasses
import mlnext
import eTaPR_pkg as etapr
import tqdm

def metrics(
    y_score: np.ndarray,
    y: np.ndarray,
    thresholds : T.List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> T.Dict[str, T.Any]:
    """Calculates performance metrics for two thresholds: yielding the
    best possible F1 and the custom threshold ``threshold``.

    Args:
        y_score (np.ndarray): Anomaly score.
        y (np.ndarray): Labels.

    Returns:
        T.Dict[str, T.Any]: Returns the metrics for a two thresholds.
    """
    # pr curve
    curve = mlnext.pr_curve(y_true=y, y_score=y_score)

    # find threshold index for best F1
    best = nanargmax(curve.f1)
    print(curve.f1)

    return {
        # 'curve': curve,
        'auc_pr': curve.auc,
        'auc_roc': roc_auc_score(y, y_score),
        'best': _calculate_metrics(y, y_score, curve, best),
        **{
            f'{t}': _calculate_metrics(
                y,
                y_score,
                curve,
                nanargmin(np.abs(curve.thresholds - t))
            )
            for t in tqdm.tqdm(thresholds)
        }
    }

def _calculate_metrics(
    y: np.ndarray,
    y_score: np.ndarray,
    curve: PRCurve,
    threshold_index: int
) -> T.Dict[str, T.Union[float, T.Dict[str, float]]]:
    """Calculates the performance metrics for a threshold index.

    Args:
        curve (PRCurve): PR Curve.
        threshold_index (int): Threshold index.

    Returns:
        T.Dict[str, float]: Returns the results.
    """
    threshold = curve.thresholds[threshold_index]
    y_hat = mlnext.apply_threshold(y_score, threshold=threshold)

    return {
        'threshold': threshold,
        'segments': recall_segments(y_hat, y),
        # 'auc_recall_segments': auc_recall_segments(y_hat, y),
        **dataclasses.asdict(curve[threshold_index]),
        **curve[threshold_index].metrics(),
        **_etarp(y, y_hat),
    }

def _etarp(
    y: np.ndarray,
    y_hat: np.ndarray
) -> T.Dict[str, T.Dict[str, float]]:
    if y.any() == 0:
        return {}

    results = etapr.evaluate_w_streams(y, y_hat, theta_p=0.5)
    results['Detected_Anomalies'] = len(results['Detected_Anomalies'])
    results['Correct_Predictions'] = len(results['Correct_Predictions'])

    return {'etarp': results}

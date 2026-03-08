"""Common evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix


def regression_metrics(y_true, y_pred):
    """Standard in-sample regression metrics: RMSE and R²."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2':   r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred):
    """Standard classification metrics: accuracy and confusion matrix."""
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'confusion': confusion_matrix(y_true, y_pred),
    }


def compute_oos_r2(y_true, y_pred, benchmark):
    """
    Out-of-sample R² relative to a benchmark prediction.

    OOS R² = 1 - SS_res / SS_tot

    where SS_tot is computed against `benchmark` (typically the training-set
    mean), not the test-set mean.  This is the standard definition in the
    return-predictability literature (Campbell & Thompson 2008).

    A value of 0 means the model is no better than predicting `benchmark`
    for every observation.  Negative values mean the model is worse.

    Parameters
    ----------
    y_true     : array-like of realized values
    y_pred     : array-like of model predictions
    benchmark  : scalar (e.g. training mean) or array-like of benchmark preds

    Returns
    -------
    float
    """
    y_true    = np.asarray(y_true, dtype=float)
    y_pred    = np.asarray(y_pred, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - benchmark) ** 2)
    return float(1.0 - ss_res / ss_tot)


def evaluate_splits(y_train, y_train_pred, y_val, y_val_pred,
                    y_test, y_test_pred, train_mean):
    """
    Compute R² and OOS R² for all three splits in one call.

    Parameters
    ----------
    y_train, y_train_pred : train actuals and predictions
    y_val,   y_val_pred   : validation actuals and predictions
    y_test,  y_test_pred  : test actuals and predictions
    train_mean            : scalar benchmark (typically y_train.mean())

    Returns
    -------
    dict with keys: r2_train, r2_val, r2_test, oos_r2_val, oos_r2_test
    """
    return {
        'r2_train':    r2_score(y_train, y_train_pred),
        'r2_val':      r2_score(y_val,   y_val_pred),
        'r2_test':     r2_score(y_test,  y_test_pred),
        'oos_r2_val':  compute_oos_r2(y_val,  y_val_pred,  train_mean),
        'oos_r2_test': compute_oos_r2(y_test, y_test_pred, train_mean),
    }

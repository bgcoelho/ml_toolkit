"""Data loading and preprocessing helpers."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_csv(path, **kwargs):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def load_parquet(path, **kwargs):
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path, **kwargs)


def time_series_split(df, date_col, train_years, val_years, date_format='yyyymm'):
    """
    Create boolean train/validation/test masks for panel data using year-based splits.

    Training  : first `train_years` years of data.
    Validation: next  `val_years`   years.
    Test      : everything after.

    Parameters
    ----------
    df          : pd.DataFrame
    date_col    : str  — column containing the date (integer yyyymm or datetime)
    train_years : int
    val_years   : int
    date_format : 'yyyymm' (integer) or 'datetime'

    Returns
    -------
    train_mask, val_mask, test_mask : boolean pd.Series
    split_info : dict with period boundaries and observation counts
    """
    if date_format == 'yyyymm':
        year_col = pd.to_numeric(df[date_col], errors='coerce') // 100
    else:
        year_col = pd.to_datetime(df[date_col]).dt.year

    start_year  = year_col.min()
    train_end   = start_year + train_years - 1
    val_end     = train_end + val_years

    train_mask = year_col <= train_end
    val_mask   = (year_col > train_end) & (year_col <= val_end)
    test_mask  = year_col > val_end

    split_info = {
        'train_start': int(start_year),
        'train_end':   int(train_end),
        'val_start':   int(train_end + 1),
        'val_end':     int(val_end),
        'test_start':  int(val_end + 1),
        'test_end':    int(year_col.max()),
        'train_n_obs': int(train_mask.sum()),
        'val_n_obs':   int(val_mask.sum()),
        'test_n_obs':  int(test_mask.sum()),
    }
    return train_mask, val_mask, test_mask, split_info


def prepare_panel_data(df, target_col, exclude_cols=None, fill_value=0):
    """
    Extract a feature matrix X and target series y from a panel DataFrame.

    All numeric columns not in `exclude_cols` or `target_col` become features.
    Missing feature values are filled with `fill_value`.
    Target is coerced to numeric (non-parseable entries become NaN).

    Parameters
    ----------
    df           : pd.DataFrame
    target_col   : str
    exclude_cols : list of str (e.g. ['yyyymm', 'permno'])
    fill_value   : scalar (default 0)

    Returns
    -------
    X            : pd.DataFrame of features
    y            : pd.Series of target values
    feature_cols : list of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []
    non_feature = set(exclude_cols) | {target_col}

    feature_cols = [
        col for col in df.columns
        if col not in non_feature and pd.api.types.is_numeric_dtype(df[col])
    ]

    X = df[feature_cols].fillna(fill_value)
    y = pd.to_numeric(df[target_col], errors='coerce')

    return X, y, feature_cols


def standardize_splits(X_train, X_val, X_test=None):
    """
    Fit a StandardScaler on X_train and apply it to all splits.

    Prevents data leakage: the scaler is never exposed to val/test statistics.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
    X_test         : pd.DataFrame or None

    Returns
    -------
    Scaled DataFrames (same index/columns) and the fitted scaler.
    If X_test is provided : X_train_s, X_val_s, X_test_s, scaler
    Otherwise             : X_train_s, X_val_s, scaler
    """
    scaler = StandardScaler()

    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )

    if X_test is not None:
        X_test_s = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        return X_train_s, X_val_s, X_test_s, scaler

    return X_train_s, X_val_s, scaler


def drop_na_targets(*splits):
    """
    Drop rows where the target y is NaN from each (X, y) pair.

    Usage:
        (X_train, y_train), (X_val, y_val) = drop_na_targets(
            (X_train, y_train), (X_val, y_val)
        )
    """
    cleaned = []
    for X, y in splits:
        valid = ~y.isna()
        cleaned.append((X[valid], y[valid]))
    return cleaned

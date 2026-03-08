"""Utility functions for building, persisting, and summarizing models."""

import os
import joblib
import pandas as pd


def save_model(model, path):
    """Serialize a fitted model to disk."""
    joblib.dump(model, path)


def load_model(path):
    """Load a serialized model from disk."""
    return joblib.load(path)


def load_config(path):
    """Read a YAML configuration file into a dict."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_param_grid(config):
    """
    Convert a config dict to sklearn GridSearchCV format.

    Scalar values are wrapped in a list; lists are kept as-is.
    """
    return {k: (v if isinstance(v, list) else [v]) for k, v in config.items()}


def save_run_summary(summary_dict, output_dir, filename):
    """
    Save a model-run summary dict as a single-row CSV.

    Creates `output_dir` if it does not exist.

    Parameters
    ----------
    summary_dict : dict   — metrics, parameters, and metadata for this run
    output_dir   : str    — directory to write the file
    filename     : str    — CSV filename (e.g. 'ols_summary.csv')

    Returns
    -------
    str  path of the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    pd.DataFrame([summary_dict]).to_csv(path, index=False)
    return path


def save_portfolio_compositions(df, output_dir, filename):
    """
    Save the portfolio-composition DataFrame (stock-level decile assignments) as CSV.

    Expected columns: date, id_col, decile, predicted_ret, actual_ret.

    Parameters
    ----------
    df         : pd.DataFrame
    output_dir : str
    filename   : str  (e.g. 'ols_portfolios.csv')

    Returns
    -------
    str  path of the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


def build_run_summary(model_name, description, tuning_params,
                      metrics, portfolio_stats, split_info,
                      n_features, fill_value, standardized, notebook=''):
    """
    Build the standardized summary dict used across all model notebooks.

    Parameters
    ----------
    model_name     : str    (e.g. 'OLS', 'Ridge', 'PLS Linear')
    description    : str    human-readable description
    tuning_params  : str    serialized tuning parameters (e.g. 'alpha=1.0')
    metrics        : dict   from evaluation.evaluate_splits()
    portfolio_stats: dict   from portfolio_utils.portfolio_summary()
    split_info     : dict   from data_utils.time_series_split()
    n_features     : int
    fill_value     : scalar  missing-value fill used during preprocessing
    standardized   : bool
    notebook       : str    (optional) source notebook name

    Returns
    -------
    dict  ready to pass to save_run_summary()
    """
    return {
        'model':             model_name,
        'description':       description,
        'tuning_parameters': tuning_params,
        'notebook':          notebook,
        # evaluation
        'train_r2':          metrics.get('r2_train'),
        'val_r2':            metrics.get('r2_val'),
        'test_r2':           metrics.get('r2_test'),
        'oos_r2_val':        metrics.get('oos_r2_val'),
        'oos_r2':            metrics.get('oos_r2_test'),
        # portfolio
        'sharpe_annualized': portfolio_stats.get('sharpe_annualized'),
        'mean_monthly_ret':  portfolio_stats.get('mean_period_return'),
        'std_monthly_ret':   portfolio_stats.get('std_period_return'),
        'n_months':          portfolio_stats.get('n_periods'),
        # data
        'n_characteristics': n_features,
        'missing_fill':      fill_value,
        'standardized':      standardized,
        # splits
        'train_start':       split_info.get('train_start'),
        'train_end':         split_info.get('train_end'),
        'train_n_obs':       split_info.get('train_n_obs'),
        'val_start':         split_info.get('val_start'),
        'val_end':           split_info.get('val_end'),
        'val_n_obs':         split_info.get('val_n_obs'),
        'test_start':        split_info.get('test_start'),
        'test_end':          split_info.get('test_end'),
        'test_n_obs':        split_info.get('test_n_obs'),
    }

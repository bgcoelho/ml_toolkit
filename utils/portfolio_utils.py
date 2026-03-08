"""Long-short portfolio formation and performance utilities."""

import numpy as np
import pandas as pd


def form_long_short_portfolio(df, date_col, pred_col, ret_col, n_deciles=10):
    """
    Form a monthly long-short portfolio by sorting predictions into deciles.

    Each period:
      - Stocks are ranked by `pred_col` and split into `n_deciles` buckets.
      - Long  : top decile (highest predicted return).
      - Short : bottom decile (decile 1).
      - Portfolio return = mean(long returns) - mean(short returns).

    Parameters
    ----------
    df         : pd.DataFrame with at least date_col, pred_col, ret_col columns
    date_col   : str  — period identifier (e.g. 'yyyymm')
    pred_col   : str  — predicted return used for ranking
    ret_col    : str  — realized return
    n_deciles  : int  (default 10)

    Returns
    -------
    portfolio_returns : pd.Series  monthly long-short returns indexed by date
    df_out           : input df with an added 'decile' column (1 = lowest)
    """
    df_out = df.copy()
    df_out[ret_col]  = np.array(df_out[ret_col],  dtype=float)
    df_out[pred_col] = np.array(df_out[pred_col], dtype=float)
    df_out['decile'] = df_out.groupby(date_col)[pred_col].transform(
        lambda x: pd.qcut(x, n_deciles, labels=False, duplicates='drop') + 1
    )

    max_decile = df_out.groupby(date_col)['decile'].transform('max')
    monthly_long  = df_out[df_out['decile'] == max_decile].groupby(date_col)[ret_col].mean()
    monthly_short = df_out[df_out['decile'] == 1].groupby(date_col)[ret_col].mean()

    common = monthly_long.index.intersection(monthly_short.index)
    portfolio_returns = monthly_long.loc[common] - monthly_short.loc[common]

    return portfolio_returns, df_out


def annualized_sharpe(returns, periods_per_year=12):
    """
    Annualized Sharpe ratio assuming a zero risk-free rate.

    Parameters
    ----------
    returns         : array-like of periodic returns
    periods_per_year: int  (12 for monthly, 252 for daily, etc.)

    Returns
    -------
    float
    """
    r = np.asarray(returns, dtype=float)
    std = r.std(ddof=1)
    if std == 0:
        return 0.0
    return float((r.mean() / std) * np.sqrt(periods_per_year))


def portfolio_summary(returns, periods_per_year=12):
    """
    Key performance statistics for a return series.

    Parameters
    ----------
    returns         : array-like of periodic returns
    periods_per_year: int

    Returns
    -------
    dict with mean_period_return, std_period_return, sharpe_annualized, n_periods
    """
    r = np.asarray(returns, dtype=float)
    return {
        'mean_period_return': float(r.mean()),
        'std_period_return':  float(r.std(ddof=1)),
        'sharpe_annualized':  annualized_sharpe(r, periods_per_year),
        'n_periods':          int(len(r)),
    }


def build_portfolio_df(df, date_col, id_col, pred_col, ret_col, n_deciles=10):
    """
    Convenience wrapper that returns the portfolio-composition DataFrame in the
    standardized format used across homework notebooks.

    Columns: date, <id_col>, decile, predicted_ret, actual_ret

    Parameters
    ----------
    df         : pd.DataFrame
    date_col   : str
    id_col     : str  (e.g. 'permno')
    pred_col   : str
    ret_col    : str
    n_deciles  : int

    Returns
    -------
    portfolio_returns : pd.Series  monthly long-short returns
    compositions_df  : pd.DataFrame  stock-level assignments
    """
    port_returns, df_out = form_long_short_portfolio(
        df, date_col, pred_col, ret_col, n_deciles
    )
    compositions = df_out[[date_col, id_col, 'decile', pred_col, ret_col]].copy()
    compositions = compositions.rename(columns={pred_col: 'predicted_ret', ret_col: 'actual_ret'})

    return port_returns, compositions

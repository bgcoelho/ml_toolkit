"""ML Toolkit utility modules.

Quick-import shortcuts:

    from utils import load_parquet, time_series_split, prepare_panel_data
    from utils import compute_oos_r2, evaluate_splits
    from utils import form_long_short_portfolio, annualized_sharpe, portfolio_summary
    from utils import build_run_summary, save_run_summary, save_portfolio_compositions
"""

from .data_utils import (
    load_csv,
    load_parquet,
    time_series_split,
    prepare_panel_data,
    standardize_splits,
    drop_na_targets,
)

from .evaluation import (
    regression_metrics,
    classification_metrics,
    compute_oos_r2,
    evaluate_splits,
)

from .portfolio_utils import (
    form_long_short_portfolio,
    annualized_sharpe,
    portfolio_summary,
    build_portfolio_df,
)

from .model_utils import (
    save_model,
    load_model,
    load_config,
    make_param_grid,
    save_run_summary,
    save_portfolio_compositions,
    build_run_summary,
)

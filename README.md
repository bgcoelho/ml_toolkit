# Machine Learning Toolkit

A reusable collection of notebooks and utilities for common machine learning tasks.

## Structure

```
ml_toolkit/
├── utils/                   # Shared Python helpers (import from any notebook)
│   ├── data_utils.py        # load_csv, load_parquet, time_series_split, prepare_panel_data, standardize_splits, drop_na_targets
│   ├── evaluation.py        # compute_oos_r2, evaluate_splits, regression_metrics, classification_metrics
│   ├── portfolio_utils.py   # form_long_short_portfolio, annualized_sharpe, portfolio_summary, build_portfolio_df
│   └── model_utils.py       # save/load model, load_config, make_param_grid, build_run_summary, save_run_summary, save_portfolio_compositions
├── models/                  # Self-contained model notebooks
│   ├── supervised/          # ridge, lasso, ridge_lasso, elasticnet, pcr_regression, pls_regression,
│   │                        # rbf_regression, kernel_ridge, gbr_regression, random_forest,
│   │                        # neural_network, logistic_classification
│   │   └── results/         # Per-model summary and portfolio CSVs (auto-generated)
│   ├── unsupervised/        # pca
│   ├── timeseries/          # forecasting, expanding_window
│   └── nlp/                 # text_lasso, tfidf_lasso, embeddings_ml, lda_topics
└── data/                    # Sample datasets (FREDMD.csv, largeml.pq, smallml.pq, gw.csv)
```

## Usage

Each model notebook is self-contained. To adapt it to a new dataset, edit the `CONFIG` dict at the top — no other changes needed.

```python
CONFIG = {
    'DATA_FILE':    'path/to/data.pq',
    'DATE_COL':     'yyyymm',
    'ID_COL':       'permno',
    'TARGET_COL':   'ret',
    'TRAIN_YEARS':  20,
    'VAL_YEARS':    12,
    ...
}
```

## Importing utils

From any notebook inside `models/`:

```python
import sys
sys.path.insert(0, '../../')

from utils import (
    load_csv, load_parquet, time_series_split, prepare_panel_data, standardize_splits, drop_na_targets,
    compute_oos_r2, evaluate_splits, regression_metrics, classification_metrics,
    form_long_short_portfolio, annualized_sharpe, portfolio_summary, build_portfolio_df,
    load_config, make_param_grid, build_run_summary, save_run_summary, save_portfolio_compositions,
)
```

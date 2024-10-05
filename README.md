# Regression of Used Car Prices

<https://www.kaggle.com/competitions/playground-series-s4e9>

## Installation

- Python3 (testd with 3.10)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -U -r requirements.txt
```

## Various Approaches

Private Score: 63426.73748 Public score: 72393.60863

- add CatBoostRegressor

Private Score: 63807.34707 Public score: 72698.31823

- add CatBoostClassifier

Private Score: 63822.27010 Public score: 72717.78853

- bin_price
- target encoding (median)
- out-of-fold features (LGBMRegressor, XGBRegressor)
- ensemble (Ridge)

TODO:

1. enrich out-of-fold features with more regressors
2. HP-tunning

#!/usr/bin/env python

# In[ ]:


from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level="INFO")
# log = logging.getLogger(__file__)
log = logging.getLogger("notebook")

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

test_id_list = list(test_df["id"])


# In[ ]:


# feature engineering
def feature_engineering(df):
    df = df.copy()

    # Simplified transmission types into two categories:
    df["transmission"] = df["transmission"].replace({"Automatic": "A/T", "Manual": "M/T"})

    # identify luxury brands
    luxury_brands = {
        'Aston',
        'Audi',
        'BMW',
        'Bentley',
        'Ferrari',
        'Jaguar',
        'Lamborghini',
        'Land',
        'Lexus',
        'Maserati',
        'Maybach',
        'McLaren',
        'Mercedes-Benz',
        'Porsche',
        'Rolls-Royce',
    }
    df["is_luxury"] = df["brand"].apply(lambda x: 1 if x in luxury_brands else 0)

    # Normalize a car’s mileage by its age.
    # This reflects how intensively the car has been used, which can affect its price.
    df["car_age"] = 2024 - df["model_year"]
    df["milage_per_year"] = df["milage"] / (df["car_age"] + 1)

    # Create cross features
    df["brand_model"] = df["brand"] + "_" + df["model"]
    df["int_ext_col"] = df["int_col"] + "_" + df["ext_col"]

    df[["fuel_type", "accident", "clean_title"]] = df[["fuel_type", "accident", "clean_title"]].fillna(
        "unknown"
    )

    df = df.drop(columns=["id"])

    return df


log.info("Feature Engineering: train_df")
train_df = feature_engineering(train_df)
log.info("Feature Engineering: test_df")
test_df = feature_engineering(test_df)


# Outlier Detection
# The car price distribution is highly right-skewed.
# This skewed distribution can negatively affect the performance of the model,
# as extreme values (outliers) tend to distort the predictions.
# To address this, use an outlier detection function based on IQR.
def bin_price(data):
    df = data.copy()
    q1 = np.percentile(df["price"], 25)
    q3 = np.percentile(df["price"], 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    df["price_bin"] = (df["price"] < upper_bound).astype(int)
    return df


log.info("Bin Price: train_df")
train_df = bin_price(train_df)


# In[ ]:


# Target Encoding
# Convert the categorical features into numerical values, representing the median price of cars for each category.
def target_encode(train_df, test_df, target_col, cat_cols, n_folds=5):
    for col in cat_cols:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=9999)
        global_median = train_df[target_col].median()

        train_df[f"{col}_te"] = np.nan
        test_df[f"{col}_te"] = np.nan

        test_col_median = train_df.groupby(col)[target_col].median()
        test_df[f"{col}_te"] = test_df[col].map(test_col_median).fillna(global_median)

        for train_idx, val_idx in kf.split(train_df):
            X_train = train_df.iloc[train_idx]
            X_val = train_df.iloc[val_idx]
            col_median = X_train.groupby(col)[target_col].median()
            train_df.loc[val_idx, f"{col}_te"] = X_val[col].map(col_median).fillna(global_median)
    return train_df, test_df


log.info("Target Encoding: train_df, test_df")
cat_cols = ["brand", "model", "brand_model", "transmission", "fuel_type", "engine"]
train_df, test_df = target_encode(train_df, test_df, "price", cat_cols)


# In[ ]:


# Model Stacking with Out-of-Fold Predictions
# The OOF predictions allow the meta-learner to make better final predictions by
# leveraging the strengths of each base model.
def get_oof_predictions(model, train_df, test_df, features, target_col, n_folds=5):
    oof_train = np.zeros(len(train_df))
    oof_test = np.zeros(len(test_df))
    oof_test_skf = np.empty((n_folds, len(test_df)))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=9999)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        X_train = train_df.iloc[train_idx][features]
        X_val = train_df.iloc[val_idx][features]
        y_train = train_df.iloc[train_idx][target_col]
        y_val = train_df.iloc[val_idx][target_col]
        model.fit(X_train, y_train)
        oof_train[val_idx] = model.predict(X_val)
        oof_test_skf[i, :] = model.predict(test_df[features])

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test


cat_features = [
    "brand",
    "model",
    "brand_model",
    "fuel_type",
    "engine",
    "transmission",
    "int_ext_col",
    # ext_col,
    # int_col,
    # accident,
    # clean_title,
]

num_features = [
    "brand_te",
    "model_te",
    "transmission_te",
    "fuel_type_te",
    "engine_te",
    "milage",
    "car_age",
    "milage_per_year",
    "is_luxury",
]

log.info("OOF Predictions: CatBoostClassifier")
catboost_clf_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 10,
    "l2_leaf_reg": 17,
    "random_strength": 11,
    # "subsample": 0.95,
    "verbose": 1,
    "cat_features": cat_features,
    "random_seed": 9999,
}
catboost_clf = CatBoostClassifier(**catboost_clf_params)
catboost_clf_oof_train, catboost_clf_oof_test = get_oof_predictions(
    catboost_clf, train_df, test_df, cat_features + num_features, "price_bin"
)

log.info("OOF Predictions: CatBoostRegressor")
catboost_reg_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 10,
    "l2_leaf_reg": 17,
    "random_strength": 11,
    "subsample": 0.95,
    "verbose": 1,
    "cat_features": ["brand", "model", "brand_model"],
    "random_seed": 9999,
    "loss_function": "RMSE",
}
catboost_reg = CatBoostRegressor(**catboost_reg_params)
catboost_reg_oof_train, catboost_reg_oof_test = get_oof_predictions(
    catboost_reg, train_df, test_df, ["brand", "model", "brand_model"] + num_features, "price"
)

log.info("OOF Predictions: LGBMRegressor")
lgbm = LGBMRegressor(
    max_depth=10,
    learning_rate=0.03,
    n_estimators=1000,
    verbose=1,
    random_state=9999,
    objective="regression",
    metric="rmse",
)
lgbm_oof_train, lgbm_oof_test = get_oof_predictions(lgbm, train_df, test_df, num_features, "price")

log.info("OOF Predictions: XGBRegressor")
xgb = XGBRegressor(
    max_depth=10, learning_rate=0.03, n_estimators=1000, random_state=9999, objective='reg:squarederror'
)
xgb_oof_train, xgb_oof_test = get_oof_predictions(xgb, train_df, test_df, num_features, "price")

# Final Ensemble Model
ensemble_train = np.column_stack(
    [catboost_clf_oof_train, catboost_reg_oof_train, lgbm_oof_train, xgb_oof_train]
)
ensemble_test = np.column_stack([catboost_clf_oof_test, catboost_reg_oof_test, lgbm_oof_test, xgb_oof_test])

log.info("Meta-Learner: Ridge")
ridge = Ridge(random_state=9999)
ridge.fit(ensemble_train, train_df["price"])
final_predictions = ridge.predict(ensemble_test)
# final_predictions = np.expm1(final_predictions_log)


# In[ ]:


submission = pd.DataFrame({"id": test_id_list, "price": final_predictions})
submission.to_csv("./data/submission.csv", index=False)

log.info("Done")

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from xgboost import XGBRegressor

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level="INFO")
log = logging.getLogger(__file__)

# 讀取資料
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

test_id_list = list(test_df["id"])


# 1. 特徵工程
def feature_engineering(df):
    df = df.copy()

    # 將自動和手動變速器簡化為 "A/T" 和 "M/T"
    df["transmission"] = df["transmission"].replace({"Automatic": "A/T", "Manual": "M/T"})

    # 提取豪華品牌
    luxury_brands = ["BMW", "Mercedes-Benz", "Audi", "Lexus"]
    df["is_luxury"] = df["brand"].apply(lambda x: 1 if x in luxury_brands else 0)

    # 創建里程數/年
    df["car_age"] = 2024 - df["model_year"]
    df["milage_per_year"] = df["milage"] / (df["car_age"] + 1)

    # 創建特徵交叉
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


# 2. 異常檢測
def bin_price(data):
    df = data.copy()
    # 計算四分位距
    Q1 = np.percentile(df["price"], 25)
    Q3 = np.percentile(df["price"], 75)
    IQR = Q3 - Q1

    # 定義異常值範圍
    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 標記異常價格
    df["price_bin"] = (df["price"] < upper_bound).astype(int)
    return df


log.info("Bin Price: train_df")
train_df = bin_price(train_df)


# 3. 目標編碼 (使用中位數)
def target_encode(train_df, test_df, target_col, cat_cols, n_folds=5):
    for col in cat_cols:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=9999)

        # 全局中位數作為回退值
        global_median = train_df[target_col].median()

        train_df[f"{col}_te"] = np.nan
        test_df[f"{col}_te"] = np.nan

        # 測試集使用整個訓練集的目標編碼
        test_col_mean = train_df.groupby(col)[target_col].median()
        test_df[f"{col}_te"] = test_df[col].map(test_col_mean).fillna(global_median)

        # 針對每個 KFold 折，進行目標編碼
        for train_idx, val_idx in kf.split(train_df):
            X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
            # 針對訓練集進行 groupby，計算每個類別的目標變數中位數
            col_mean = X_train.groupby(col)[target_col].median()
            # 將中位數應用於驗證集，對於未出現的類別使用全局中位數作為回退
            train_df.loc[val_idx, f"{col}_te"] = X_val[col].map(col_mean).fillna(global_median)

    return train_df, test_df


log.info("Target Encoding: train_df, test_df")
cat_cols = ["brand", "model", "transmission", "fuel_type", "engine"]
train_df, test_df = target_encode(train_df, test_df, "price", cat_cols)

# print(train_df.isna().sum())
# print(test_df.isna().sum())
# raise

# 4. 模型訓練與預測
# 定義模型
# svr = SVR(kernel="rbf")
catboost_clf_params = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 10,
    "l2_leaf_reg": 17,
    "random_strength": 11,
    "subsample": 0.95,
    "verbose": 0,
    "cat_features": cat_cols,
    "random_seed": 9999,
}
catboost_clf = CatBoostClassifier(**catboost_clf_params)
lgbm = LGBMRegressor(max_depth=10, learning_rate=0.03, n_estimators=1000, random_state=9999)
xgb = XGBRegressor(max_depth=10, learning_rate=0.03, n_estimators=1000, random_state=9999)
ridge = Ridge(random_state=9999)


# OOF 預測
def get_oof_predictions(model, train_df, test_df, features, target_col, n_folds=5):
    # print(train_df.columns)
    # Index(['id', 'brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine',
    #     'transmission', 'ext_col', 'int_col', 'accident', 'clean_title',
    #     'price', 'is_luxury', 'car_age', 'milage_per_year', 'brand_model',
    #     'int_ext_col', 'price_bin', 'brand_te', 'model_te', 'transmission_te',
    #     'fuel_type_te', 'engine_te'],
    #     dtype='object')
    oof_train = np.zeros(len(train_df))
    oof_test = np.zeros(len(test_df))
    oof_test_skf = np.empty((n_folds, len(test_df)))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=9999)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        X_train, X_val = train_df.iloc[train_idx][features], train_df.iloc[val_idx][features]
        y_train, y_val = train_df.iloc[train_idx][target_col], train_df.iloc[val_idx][target_col]

        model.fit(X_train, y_train)
        oof_train[val_idx] = model.predict(X_val)
        oof_test_skf[i, :] = model.predict(test_df[features])

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test


features = [
    "brand_te",
    "model_te",
    "transmission_te",
    "fuel_type_te",
    "engine_te",
    "milage",
    "car_age",
    "milage_per_year",
]

# svr_oof_train, svr_oof_test = get_oof_predictions(svr, train_df, test_df, features, "price")
# log.info("Get OOF Predictions: catboost_clf")
# catboost_oof_train, catboost_oof_test = get_oof_predictions(catboost_clf, train_df, test_df, features, "price_bin")
log.info("Get OOF Predictions: lgbm")
lgbm_oof_train, lgbm_oof_test = get_oof_predictions(lgbm, train_df, test_df, features, "price")
log.info("Get OOF Predictions: xgb")
xgb_oof_train, xgb_oof_test = get_oof_predictions(xgb, train_df, test_df, features, "price")

# 5. 集成模型 (Ridge 回歸)
# NOTE: drop svr_oof_train(test), catboost
ensemble_train = np.column_stack([lgbm_oof_train, xgb_oof_train])
ensemble_test = np.column_stack([lgbm_oof_test, xgb_oof_test])

log.info("Ridge Regression")
ridge.fit(ensemble_train, train_df["price"])
final_predictions = ridge.predict(ensemble_test)

# 6. 儲存預測結果
submission = pd.DataFrame({"id": test_id_list, "price": final_predictions})
submission.to_csv("./data/submission.csv", index=False)

log.info("Done")

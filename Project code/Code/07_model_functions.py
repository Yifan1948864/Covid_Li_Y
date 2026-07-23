import numpy as np
import pandas as pd
import optuna

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


# Read train/test data and separate the outcome
def load_train_test_data(train_file, test_file, target):
    train_data = pd.read_csv(f"../Train and test/{train_file}")
    test_data = pd.read_csv(f"../Train and test/{test_file}")

    X_train = train_data.drop(columns=[target])
    y_train = train_data[target]

    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    return X_train, y_train, X_test, y_test


# Tune a Random Forest using training data only
def tune_random_forest(
    X_train,
    y_train,
    n_estimators=500,
    n_trials=200,
    random_state=3034,
):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=trial.suggest_int("max_depth", 10, 40),
            min_samples_split=trial.suggest_int(
                "min_samples_split", 2, 10
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5, None]
            ),
            random_state=random_state,
            n_jobs=1,
        )

        auc_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        return auc_scores.mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    return study


# Fit a Random Forest using the selected parameters
def fit_random_forest(
    X_train,
    y_train,
    best_params,
    n_estimators=500,
    random_state=3034,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        **best_params,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model


# Repeat model fitting to estimate stable feature importance
def repeated_feature_importance(
    model,
    X_train,
    y_train,
    n_repeats=100,
    random_state=3034,
):
    importance_list = []

    for i in range(n_repeats):
        repeated_model = clone(model)
        repeated_model.set_params(random_state=random_state + i)
        repeated_model.fit(X_train, y_train)
        importance_list.append(repeated_model.feature_importances_)

    importance_array = np.array(importance_list)

    importance_results = pd.DataFrame({
        "feature": X_train.columns,
        "median_importance": np.median(importance_array, axis=0),
        "q1": np.percentile(importance_array, 25, axis=0),
        "q3": np.percentile(importance_array, 75, axis=0),
    })
    importance_results["iqr"] = (
        importance_results["q3"] - importance_results["q1"]
    )
    importance_results = importance_results.sort_values(
        by="median_importance",
        ascending=False,
    )

    return importance_results


# Tune XGBoost using training data only
def tune_xgboost(
    X_train,
    y_train,
    n_estimators,
    n_trials=200,
    random_state=3034,
):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    def objective(trial):
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
        )

        auc_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        return auc_scores.mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    return study


# Fit XGBoost using the selected parameters
def fit_xgboost(
    X_train,
    y_train,
    best_params,
    n_estimators,
    random_state=3034,
):
    model = XGBClassifier(
        n_estimators=n_estimators,
        **best_params,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model


# Evaluate a fitted binary classification model
def evaluate_binary_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }

    return pd.Series(results)
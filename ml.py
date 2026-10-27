"""Data loading, model training and prediction (scikit-learn)."""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from simulator import MAX_MINUTES, generate_sessions, sim_time

BASE = Path(__file__).parent
DEFAULT_CSV = BASE / "data" / "ev_sessions.csv"

# To use your own dataset: set the EV_DATA_CSV environment variable to a CSV file path.
CUSTOM_CSV = os.environ.get("EV_DATA_CSV")
DATA_PATH = Path(CUSTOM_CSV) if CUSTOM_CSV else DEFAULT_CSV
IS_SIMULATED = not CUSTOM_CSV

RAW_COLUMNS = ["battery_kwh", "start_soc", "target_soc", "charger_kw", "temp_c"]
TARGET = "charging_time_min"
FEATURES = RAW_COLUMNS + ["energy_kwh"]

LABELS = {
    "battery_kwh": ("Battery capacity", "kWh"),
    "start_soc": ("Starting charge", "%"),
    "target_soc": ("Target charge", "%"),
    "charger_kw": ("Charger power", "kW"),
    "temp_c": ("Temperature", "°C"),
    "energy_kwh": ("Energy to add", "kWh"),
}


def add_features(df):
    df = df.copy()
    df["energy_kwh"] = df["battery_kwh"] * (df["target_soc"] - df["start_soc"]) / 100
    return df


@lru_cache(maxsize=1)
def load_data():
    if IS_SIMULATED and not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        generate_sessions().to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in RAW_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{DATA_PATH.name} is missing columns: {', '.join(missing)}")
    return add_features(df[RAW_COLUMNS + [TARGET]].dropna())


@lru_cache(maxsize=1)
def split():
    df = load_data()
    return train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)


@lru_cache(maxsize=1)
def linear_model():
    X_train, _, y_train, _ = split()
    return LinearRegression().fit(X_train, y_train)


@lru_cache(maxsize=64)
def tree_model(depth, leaf):
    X_train, _, y_train, _ = split()
    return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=42).fit(X_train, y_train)


def scores(model):
    X_train, X_test, y_train, y_test = split()
    p_test, p_train = model.predict(X_test), model.predict(X_train)
    return {
        "mae": float(mean_absolute_error(y_test, p_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, p_test))),
        "r2_test": float(r2_score(y_test, p_test)),
        "r2_train": float(r2_score(y_train, p_train)),
    }


def summary(depth, leaf):
    X_train, X_test, y_train, y_test = split()
    lr, dt = linear_model(), tree_model(depth, leaf)

    std = X_train.std(ddof=0).values
    lr_coef = []
    for i, name in enumerate(FEATURES):
        label, unit = LABELS[name]
        lr_coef.append({
            "label": label, "unit": unit,
            "standardized": float(lr.coef_[i] * std[i]),
            "per_unit": float(lr.coef_[i]),
        })
    dt_importance = [
        {"label": LABELS[name][0], "value": float(v)}
        for name, v in zip(FEATURES, dt.feature_importances_)
    ]
    return {
        "simulated": IS_SIMULATED,
        "data_file": DATA_PATH.name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "lr": scores(lr),
        "dt": scores(dt),
        "actual": [round(float(v), 1) for v in y_test],
        "lr_pred": [round(float(v), 1) for v in lr.predict(X_test)],
        "dt_pred": [round(float(v), 1) for v in dt.predict(X_test)],
        "lr_coef": lr_coef,
        "dt_importance": dt_importance,
        "dt_leaves": int(dt.get_n_leaves()),
        "dt_depth": int(dt.get_depth()),
    }


def predict(battery, start, target, charger, temp, depth, leaf):
    row = add_features(pd.DataFrame(
        [[battery, start, target, charger, temp]], columns=RAW_COLUMNS))[FEATURES]
    lr, dt = linear_model(), tree_model(depth, leaf)

    lr_min = max(1.0, float(lr.predict(row)[0]))
    dt_min = float(dt.predict(row)[0])

    # Yes/no questions the tree asked for this session.
    tree, x = dt.tree_, row.values[0]
    path, leaf_id = [], 0
    for node in dt.decision_path(row).indices:
        if tree.children_left[node] == tree.children_right[node]:
            leaf_id = int(node)
            break
        f = int(tree.feature[node])
        label, unit = LABELS[FEATURES[f]]
        path.append({
            "label": label, "unit": unit,
            "threshold": float(tree.threshold[node]),
            "left": bool(x[f] <= tree.threshold[node]),
        })

    sim = sim_time(battery, start, target, charger, temp) if IS_SIMULATED else None
    return {
        "lr_min": lr_min,
        "dt_min": dt_min,
        "sim_min": sim,
        "beyond_range": bool(sim is not None and sim > MAX_MINUTES),
        "path": path,
        "leaf_samples": int(tree.n_node_samples[leaf_id]),
        "leaf_value": float(tree.value[leaf_id][0][0]),
    }

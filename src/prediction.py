"""
prediction.py
-------------
Clean model prediction abstraction layer.
Handles loading, caching, and inference for all three disease models.
Does NOT modify training — inference only.
"""

from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Lazy imports for optional DL deps
_keras = None

def _get_keras():
    global _keras
    if _keras is None:
        from tensorflow import keras
        _keras = keras
    return _keras


# ---------------------------------------------------------------------------
# Model registry (populated at startup by load_all_models)
# ---------------------------------------------------------------------------
_registry: dict = {
    "heart":    {"model": None, "scaler": None, "type": "rf"},
    "diabetes": {"model": None, "scaler": None, "type": "ann"},
    "parkinson":{"model": None, "scaler": None, "type": "ann"},
}


def load_all_models():
    """Load all pre-trained models from disk. Call once at app startup."""
    keras = _get_keras()

    # Heart — Random Forest (trained on the fly from CSV, no .pkl)
    # Kept as None; predict_heart() handles training inline like original app.py

    # Diabetes ANN
    diabetes_model_path = os.path.join(MODELS_DIR, "diabetes_ann_model.h5")
    diabetes_scaler_path = os.path.join(MODELS_DIR, "diabetes_scaler.pkl")
    if os.path.exists(diabetes_model_path):
        _registry["diabetes"]["model"] = keras.models.load_model(diabetes_model_path)
    if os.path.exists(diabetes_scaler_path):
        with open(diabetes_scaler_path, "rb") as f:
            _registry["diabetes"]["scaler"] = pickle.load(f)

    # Parkinson ANN
    parkinson_model_path = os.path.join(MODELS_DIR, "parkinson_ann_model.h5")
    parkinson_scaler_path = os.path.join(MODELS_DIR, "parkinson_scaler.pkl")
    if os.path.exists(parkinson_model_path):
        _registry["parkinson"]["model"] = keras.models.load_model(parkinson_model_path)
    if os.path.exists(parkinson_scaler_path):
        with open(parkinson_scaler_path, "rb") as f:
            _registry["parkinson"]["scaler"] = pickle.load(f)

    return _registry


def predict_heart(features: dict) -> tuple[int, float]:
    """
    Predict heart disease using Random Forest trained on heart.csv.
    features: dict matching heart.csv column names (excluding 'target').

    Returns: (prediction_int, base_probability)
    """
    from sklearn.ensemble import RandomForestClassifier

    heart_csv = os.path.join(DATA_DIR, "heart.csv")
    if not os.path.exists(heart_csv):
        raise FileNotFoundError("heart.csv not found")

    data = pd.read_csv(heart_csv)
    X = data.drop("target", axis=1)
    y = data["target"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    df = pd.DataFrame([features])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return pred, prob


def predict_diabetes(features: list) -> tuple[int, float]:
    """
    Predict diabetes using saved ANN model.
    features: list of [gender, age, hypertension, heart_disease, bmi, hba1c, glucose]

    Returns: (prediction_int, base_probability)
    """
    m = _registry["diabetes"]["model"]
    s = _registry["diabetes"]["scaler"]
    if m is None or s is None:
        raise RuntimeError("Diabetes model not loaded")

    x = np.array([features], dtype=float)
    x_scaled = s.transform(x)
    prob = float(m.predict(x_scaled)[0][0])
    return int(prob >= 0.5), prob


def predict_parkinson(features: list) -> tuple[int, float]:
    """
    Predict Parkinson's using saved ANN model.
    features: list of 22 acoustic voice measures

    Returns: (prediction_int, base_probability)
    """
    m = _registry["parkinson"]["model"]
    s = _registry["parkinson"]["scaler"]
    if m is None or s is None:
        raise RuntimeError("Parkinson model not loaded")

    x = np.array([features], dtype=float)
    x_scaled = s.transform(x)
    prob = float(m.predict(x_scaled)[0][0])
    return int(prob >= 0.5), prob


def get_registry():
    return _registry

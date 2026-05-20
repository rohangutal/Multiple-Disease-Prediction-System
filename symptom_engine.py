"""
symptom_engine.py
-----------------
Symptom-based scoring and matching engine for the Multi-Disease Prediction System.
Maps user-selected symptoms to disease-specific weighted profiles derived from
dataset feature importance. Produces:
  - symptom_match_score  (0.0 – 1.0)
  - matched_symptoms     (list of matched symptom names)
  - mismatch_flag        (bool – symptoms strongly contradict prediction)
"""

# ---------------------------------------------------------------------------
# Disease symptom profiles
# Each entry: symptom_label -> (weight, dataset_feature_hint, severity_tier)
#   weight        : relative clinical importance (higher = stronger indicator)
#   dataset_col   : mapped dataset column (informational)
#   severity_tier : 1=Mild, 2=Moderate, 3=High
# ---------------------------------------------------------------------------

SYMPTOM_PROFILES = {

    # ── Heart Disease ──────────────────────────────────────────────────────
    "heart": {
        "Chest pain": {
            "weight": 0.28, "dataset_col": "cp", "severity": 3,
            "description": "Chest pain / angina (cp field) — strongest indicator of CAD"
        },
        "Shortness of breath": {
            "weight": 0.22, "dataset_col": "thalach", "severity": 3,
            "description": "Exercise-induced dyspnea; inversely correlated with thalach"
        },
        "Palpitations": {
            "weight": 0.14, "dataset_col": "restecg", "severity": 2,
            "description": "Abnormal resting ECG patterns"
        },
        "Dizziness": {
            "weight": 0.14, "dataset_col": "oldpeak", "severity": 2,
            "description": "ST depression (oldpeak) reflects myocardial stress"
        },
        "Fatigue": {
            "weight": 0.12, "dataset_col": "exang", "severity": 2,
            "description": "Exercise-induced angina (exang)"
        },
        "Cold Sweats": {
            "weight": 0.10, "dataset_col": "slope", "severity": 2,
            "description": "Sympathetic activation — correlates with slope/thal"
        },
        "Jaw/Arm pain": {
            "weight": 0.10, "dataset_col": "ca", "severity": 2,
            "description": "Referred pain radiating to jaw or arm — sign of angina"
        },
    },

    # ── Diabetes ───────────────────────────────────────────────────────────
    "diabetes": {
        "Frequent urination": {
            "weight": 0.25, "dataset_col": "blood_glucose_level", "severity": 3,
            "description": "Polyuria — directly caused by elevated blood glucose"
        },
        "Excessive thirst": {
            "weight": 0.22, "dataset_col": "blood_glucose_level", "severity": 3,
            "description": "Polydipsia — osmotic consequence of hyperglycemia"
        },
        "Unexplained weight loss": {
            "weight": 0.18, "dataset_col": "bmi", "severity": 3,
            "description": "Rapid BMI drop — catabolic effect of insulin deficiency"
        },
        "Fatigue": {
            "weight": 0.14, "dataset_col": "HbA1c_level", "severity": 2,
            "description": "Chronic hyperglycemia (high HbA1c) causes persistent fatigue"
        },
        "Blurred vision": {
            "weight": 0.12, "dataset_col": "HbA1c_level", "severity": 2,
            "description": "Lens osmotic swelling at high glucose — HbA1c correlated"
        },
        "Slow-healing sores": {
            "weight": 0.09, "dataset_col": "blood_glucose_level", "severity": 2,
            "description": "Impaired wound healing from hyperglycemia & neuropathy"
        },
    },

    # ── Parkinson's ────────────────────────────────────────────────────────
    "parkinson": {
        "Tremor": {
            "weight": 0.27, "dataset_col": "MDVP:Jitter(%)", "severity": 3,
            "description": "Resting tremor — reflected in vocal jitter instability"
        },
        "Slowed movement (bradykinesia)": {
            "weight": 0.23, "dataset_col": "spread1", "severity": 3,
            "description": "Bradykinesia — captured by nonlinear spread1 measure"
        },
        "Rigid muscles": {
            "weight": 0.18, "dataset_col": "RPDE", "severity": 3,
            "description": "Muscle rigidity — correlated with RPDE dynamical complexity"
        },
        "Impaired posture and balance": {
            "weight": 0.14, "dataset_col": "DFA", "severity": 2,
            "description": "Postural instability — DFA signal fractal measure"
        },
        "Loss of automatic movements": {
            "weight": 0.11, "dataset_col": "PPE", "severity": 2,
            "description": "Hypomimia / loss of arm swing — PPE entropy measure"
        },
        "Speech changes": {
            "weight": 0.07, "dataset_col": "HNR", "severity": 1,
            "description": "Dysphonia — directly measured by HNR and shimmer"
        },
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_symptom_list(disease: str) -> list[str]:
    """Return ordered list of symptom names for a disease."""
    profile = SYMPTOM_PROFILES.get(disease, {})
    return list(profile.keys())


def score_symptoms(disease: str, selected_symptoms: list[str], predicted_positive: bool) -> dict:
    """
    Compute the symptom match score and metadata.

    Parameters
    ----------
    disease             : 'heart' | 'diabetes' | 'parkinson'
    selected_symptoms   : list of symptom labels the user checked
    predicted_positive  : True if the base model predicts disease present

    Returns
    -------
    dict with keys:
        symptom_match_score   float 0–1   (weighted match fraction)
        raw_symptom_count     int
        matched_symptoms      list[str]   symptoms that match the disease profile
        unmatched_symptoms    list[str]   symptoms selected but not in profile (rare)
        mismatch_flag         bool        True when strong symptom mismatch detected
        symptom_details       list[dict]  per-symptom detail for display
        total_weight_possible float       sum of all weights in profile
    """
    profile = SYMPTOM_PROFILES.get(disease, {})
    if not profile:
        return _empty_result()

    total_weight_possible = sum(v["weight"] for v in profile.values())
    matched = []
    unmatched = []
    weighted_score = 0.0
    details = []

    for sym in selected_symptoms:
        if sym in profile:
            info = profile[sym]
            matched.append(sym)
            weighted_score += info["weight"]
            details.append({
                "name": sym,
                "weight": info["weight"],
                "severity": info["severity"],
                "dataset_col": info["dataset_col"],
                "description": info["description"],
                "matched": True,
            })
        else:
            unmatched.append(sym)

    # Normalise to 0-1
    match_score = weighted_score / total_weight_possible if total_weight_possible > 0 else 0.0
    match_score = min(match_score, 1.0)

    # Mismatch: model says positive but zero high-severity symptoms selected (and user selected ≥2)
    high_severity_matched = sum(1 for d in details if d["severity"] == 3)
    mismatch = (predicted_positive
                and len(selected_symptoms) >= 2
                and high_severity_matched == 0
                and match_score < 0.15)

    return {
        "symptom_match_score": round(match_score, 4),
        "raw_symptom_count": len(selected_symptoms),
        "matched_symptoms": matched,
        "unmatched_symptoms": unmatched,
        "mismatch_flag": mismatch,
        "symptom_details": details,
        "total_weight_possible": round(total_weight_possible, 4),
        "weighted_score": round(weighted_score, 4),
    }


def _empty_result() -> dict:
    return {
        "symptom_match_score": 0.0,
        "raw_symptom_count": 0,
        "matched_symptoms": [],
        "unmatched_symptoms": [],
        "mismatch_flag": False,
        "symptom_details": [],
        "total_weight_possible": 1.0,
        "weighted_score": 0.0,
    }

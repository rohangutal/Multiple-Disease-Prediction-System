# HealthCure — AI Multi-Disease Prediction System (Enhanced)

## Overview

This is the enhanced version of the HealthCure Flask application, implementing
AI-driven symptom-based confidence enhancement on top of the existing pre-trained
ML models (Random Forest for Heart Disease, ANN for Diabetes & Parkinson's).

**No model retraining was performed. All enhancements are in the prediction layer.**

---

## What's New

### New Python Modules

| File | Purpose |
|------|---------|
| `symptom_engine.py` | Symptom-to-feature mapping, weighted scoring, match detection |
| `confidence_engine.py` | Confidence enhancement formula + rule-based layer |
| `prediction.py` | Clean model abstraction layer (load/predict helpers) |

### Enhanced `app.py`
- Integrates `symptom_engine` and `confidence_engine` into all three disease routes
- Added `/report/download` endpoint for HTML patient report download
- Backward-compatible — all original routes work as before

### Enhanced Templates

#### `templates/_result_panel.html` (NEW — Jinja2 Macro)
Shared result panel rendered by all three disease pages, providing:
- **Semicircle confidence gauge** (colour-coded: green/yellow/red)
- **Score breakdown bars**: Base ML + Symptom + History + Rule adjustments
- **Matched Symptoms display** with tags
- **3-tab panel**: Recommendations | AI Explanation | Symptom Details
- **Download Report button** (saves `.html` patient report)

#### `templates/heart.html`, `diabetes.html`, `parkinson.html`
- Symptom checkboxes now submit individual `sym_<name>` hidden inputs
- All symptom data passed to `symptom_engine` for weighted matching
- Legacy simple result box replaced by the rich `_result_panel.html` macro

---

## Confidence Formula

```
Final Confidence = Base_ML_Probability
                 + (Symptom_Match_Score  × W1=0.15)
                 + (History_Impact_Score × W2=0.08)
                 ± Rule_Adjustment

Rule Layer:
  IF model=Positive AND strong symptom match  → +8%
  IF model=Positive AND symptom mismatch      → -6%
  IF borderline probability AND strong match  → +5%

All scores clamped to [1%, 99%]
```

---

## Symptom Profiles (from dataset feature analysis)

### Heart Disease
| Symptom | Dataset Feature | Weight |
|---------|----------------|--------|
| Chest pain | cp | 28% |
| Shortness of breath | thalach | 22% |
| Palpitations | restecg | 14% |
| Dizziness | oldpeak | 14% |
| Fatigue | exang | 12% |
| Cold Sweats | slope/thal | 10% |

### Diabetes
| Symptom | Dataset Feature | Weight |
|---------|----------------|--------|
| Frequent urination | blood_glucose_level | 25% |
| Excessive thirst | blood_glucose_level | 22% |
| Unexplained weight loss | bmi | 18% |
| Fatigue | HbA1c_level | 14% |
| Blurred vision | HbA1c_level | 12% |
| Slow-healing sores | blood_glucose_level | 9% |

### Parkinson's
| Symptom | Dataset Feature | Weight |
|---------|----------------|--------|
| Tremor | MDVP:Jitter(%) | 27% |
| Slowed movement | spread1 | 23% |
| Rigid muscles | RPDE | 18% |
| Impaired posture | DFA | 14% |
| Loss of auto movements | PPE | 11% |
| Speech changes | HNR | 7% |

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure model files are present
#    diabetes_ann_model.h5, diabetes_scaler.pkl
#    parkinson_ann_model.h5, parkinson_scaler.pkl
#    heart.csv, diabetes.csv, parkinsons.csv

# 3. Run Flask app
python app.py

# 4. Open browser at http://127.0.0.1:5000
```

**Admin panel:** http://127.0.0.1:5000/records
Username: `admin` | Password: `admin123`

---

## Architecture

```
User Input (Form)
       │
       ▼
┌─────────────────────┐
│    app.py route     │  ← No model changes
│  (heart/diabetes/   │
│   parkinson)        │
└──────────┬──────────┘
           │
     ┌─────▼──────┐     ┌──────────────────┐
     │  ML Model  │     │  symptom_engine  │
     │ (RF / ANN) │     │ score_symptoms() │
     └─────┬──────┘     └────────┬─────────┘
           │ base_prob           │ match_score
           └──────────┬──────────┘
                      ▼
           ┌──────────────────────┐
           │  confidence_engine   │
           │  enhance_confidence()│
           │                      │
           │  Formula + Rules     │
           └──────────┬───────────┘
                      │ final_prob, risk_level,
                      │ explanation, matched_syms
                      ▼
           ┌──────────────────────┐
           │  _result_panel.html  │
           │  (Jinja2 Macro)      │
           │  Gauge + Bars + Tabs │
           └──────────────────────┘
```

---

## Constraints Respected

- ❌ Model training NOT modified
- ❌ Models NOT retrained  
- ✅ Only prediction layer enhanced
- ✅ Fully backward compatible
- ✅ All original routes preserved

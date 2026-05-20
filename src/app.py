"""
app.py  —  HealthCure: AI Multi-Disease Prediction System
Enhanced with:
  • symptom_engine.py   → symptom scoring & matching
  • confidence_engine.py → intelligent confidence enhancement
  • prediction.py       → model abstraction layer (optional)
  • PDF report download
  • Richer result dicts with explanation & matched symptoms
"""

from flask import Flask, render_template, request, flash, redirect, url_for, session, make_response
import numpy as np
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import pickle
import os
# Reduce TensorFlow/absl verbosity (info/warning spam in dev terminals)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── New modular imports ────────────────────────────────────────────────────
from src.symptom_engine import score_symptoms, get_symptom_list
from src.confidence_engine import enhance_confidence

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
INSTANCE_DIR = os.path.join(ROOT_DIR, 'instance')
INSTANCE_DB_PATH = os.path.join(INSTANCE_DIR, 'submissions.db')
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = 'smvhskcj7679e8287efuiwu'

SYMPTOMS_HEART     = get_symptom_list("heart")
SYMPTOMS_DIABETES  = get_symptom_list("diabetes")
SYMPTOMS_PARKINSON = get_symptom_list("parkinson")

app.config['ADMIN_ID']  = 'admin'
app.config['ADMIN_PWD'] = 'admin123'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{INSTANCE_DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class Submission(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(120))
    lastname  = db.Column(db.String(120))
    phone_no  = db.Column(db.String(50))
    area_code = db.Column(db.String(50))
    email     = db.Column(db.String(200))
    disease   = db.Column(db.String(50))
    inputs    = db.Column(db.Text)
    result    = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# ===========================================================================
# Helpers
# ===========================================================================

def encode_patient_history(form):
    pa_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    dq_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
    return {
        'smoking':                 int(form.get('smoking', 0)),
        'alcohol':                 int(form.get('alcohol', 0)),
        'family_history_heart':    int(form.get('family_history_heart', 0)),
        'family_history_diabetes': int(form.get('family_history_diabetes', 0)),
        'family_history_parkinson':int(form.get('family_history_parkinson', 0)),
        'stress_level':            int(form.get('stress_level', 3)),
        'physical_activity':       pa_map.get(form.get('physical_activity', 'Moderate'), 1),
        'diet_quality':            dq_map.get(form.get('diet_quality', 'Average'), 1),
    }


def _sym_key(name):
    return f"sym_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}"


def get_selected_symptoms(form, symptom_list):
    return [s for s in symptom_list if form.get(_sym_key(s)) == '1']


def build_enhanced_result(disease, base_prob, predicted_positive,
                           firstname, lastname, history, selected_symptoms,
                           family_history_key,
                           subtype_pos, subtype_neg,
                           subtype_desc_pos, subtype_desc_neg,
                           recs_pos, recs_neg):
    sym_result = score_symptoms(disease, selected_symptoms, predicted_positive)
    conf = enhance_confidence(
        base_prob=base_prob,
        predicted_positive=predicted_positive,
        symptom_result=sym_result,
        history=history,
        family_history_key=family_history_key,
    )
    final_prob   = conf["final_prob"]
    is_positive  = final_prob >= 0.5
    label_map = {
        "heart":     ("Positive", "Negative"),
        "diabetes":  ("Diabetic", "Not Diabetic"),
        "parkinson": ("Positive", "Negative"),
    }
    pos_label, neg_label = label_map.get(disease, ("Positive", "Negative"))

    return {
        "prediction":          pos_label if is_positive else neg_label,
        "is_positive":         is_positive,
        "probability":         conf["prob_percent"],
        "risk_level":          conf["risk_level"],
        "confidence_tier":     conf["confidence_tier"],
        "base_probability":    round(conf["base_prob"] * 100, 1),
        "symptom_adjustment":  round(conf["symptom_adjustment"] * 100, 1),
        "history_adjustment":  round(conf["history_adjustment"] * 100, 1),
        "rule_adjustment":     round(conf["rule_adjustment"] * 100, 1),
        "matched_symptoms":    sym_result["matched_symptoms"],
        "symptom_details":     sym_result["symptom_details"],
        "symptom_match_score": round(sym_result["symptom_match_score"] * 100, 1),
        "mismatch_flag":       sym_result["mismatch_flag"],
        "subtype":      subtype_pos if is_positive else subtype_neg,
        "subtype_desc": subtype_desc_pos if is_positive else subtype_desc_neg,
        "recommendations": recs_pos if is_positive else recs_neg,
        "explanation":     conf["explanation"],
        "firstname": firstname,
        "lastname":  lastname,
        "disease":   disease.capitalize(),
    }


# ===========================================================================
# Model loading helpers
# ===========================================================================

def _build_ann(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _train_diabetes_model():
    diabetes_csv = os.path.join(DATA_DIR, 'diabetes.csv')
    if not os.path.exists(diabetes_csv):
        return None, None
    print("[INFO] Training diabetes ANN from diabetes.csv ...")
    df = pd.read_csv(diabetes_csv)
    if df['gender'].dtype == object:
        df['gender'] = df['gender'].str.lower().map({'male':1,'female':0,'other':0})
    FEATURES = ['gender','age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level']
    X = df[FEATURES].values.astype(float); y = df['diabetes'].values.astype(float)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
    m = _build_ann(Xtr.shape[1])
    m.save(os.path.join(MODELS_DIR, 'diabetes_ann_model.h5'))
    with open(os.path.join(MODELS_DIR, 'diabetes_scaler.pkl'),'wb') as f: pickle.dump(sc, f)
    return m, sc


def _train_parkinson_model():
    csv_path = os.path.join(DATA_DIR, 'parkinsons.csv') if os.path.exists(os.path.join(DATA_DIR, 'parkinsons.csv')) else os.path.join(DATA_DIR, 'parkinson.csv')
    if not os.path.exists(csv_path):
        return None, None
    print(f"[INFO] Training Parkinson ANN from {csv_path} ...")
    df = pd.read_csv(csv_path)
    FEATURES = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
                'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
                'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR',
                'RPDE','DFA','spread1','spread2','D2','PPE']
    X = df[FEATURES].values.astype(float); y = df['status'].values.astype(float)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
    m = _build_ann(Xtr.shape[1])
    m.fit(Xtr, ytr, epochs=100, batch_size=16, validation_data=(Xv,yv),
          callbacks=[EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True,verbose=0)], verbose=0)
    m.save(os.path.join(MODELS_DIR, 'parkinson_ann_model.h5'))
    with open(os.path.join(MODELS_DIR, 'parkinson_scaler.pkl'),'wb') as f: pickle.dump(sc, f)
    return m, sc


def _load_model(filename):
    model_path = os.path.join(MODELS_DIR, filename)
    return keras.models.load_model(model_path) if os.path.exists(model_path) else None


def _load_scaler(filename):
    scaler_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None


diabetes_dl_model   = _load_model('diabetes_ann_model.h5')
diabetes_dl_scaler  = _load_scaler('diabetes_scaler.pkl')
parkinson_dl_model  = _load_model('parkinson_ann_model.h5')
parkinson_dl_scaler = _load_scaler('parkinson_scaler.pkl')


# ===========================================================================
# Chatbot
# ===========================================================================

CHAT_KB_FILE = os.path.join(INSTANCE_DIR, 'chat_kb.json')
CHAT_HISTORY_FILE = os.path.join(INSTANCE_DIR, 'chat_history.json')
DEFAULT_CHAT_KB = [
    {'question':'what are common heart disease symptoms','answer':'Common heart disease symptoms include chest pain, shortness of breath, fatigue, dizziness, and cold sweats. If you experience these, seek medical advice.'},
    {'question':'how can i prevent heart disease','answer':'Prevent heart disease with regular exercise, healthy diet, not smoking, maintaining healthy weight, and managing blood pressure, cholesterol, and diabetes.'},
    {'question':'what is a normal blood pressure','answer':'Normal blood pressure is around 120/80 mmHg. Values above 130/80 are considered elevated, and you should monitor them with a doctor.'},
    {'question':'diabetes risk factors','answer':'Diabetes risk factors include high BMI, family history, poor diet, physical inactivity, high blood pressure, and high cholesterol.'},
    {'question':'what is hba1c','answer':'HbA1c measures average blood glucose over 2-3 months; normal is below 5.7%. Higher values indicate prediabetes or diabetes risk.'},
    {'question':'what does bmi mean','answer':'BMI is body mass index, a ratio of weight to height. 18.5-24.9 is normal, 25-29.9 is overweight, and 30+ is obese.'},
    {'question':'parkinson disease signs','answer':'Parkinson signs include tremor, muscle stiffness, slow movement, and balance problems. Diagnosis and care should be done by a specialist.'},
    {'question':'what should i do if my sugar is high','answer':"If your blood sugar is high, stay hydrated, do light exercise, and follow your doctor's guidance. Avoid sugary foods and monitor levels closely."},
    {'question':'is this medical advice','answer':'This chatbot provides educational guidance only and is not a substitute for medical consultation. Always consult a healthcare professional for personal medical advice.'},
]


def _load_chat_kb():
    if os.path.exists(CHAT_KB_FILE):
        try:
            with open(CHAT_KB_FILE,'r',encoding='utf-8') as f: data=json.load(f)
            if isinstance(data,list) and all('question' in i and 'answer' in i for i in data): return data
        except Exception: pass
    with open(CHAT_KB_FILE,'w',encoding='utf-8') as f: json.dump(DEFAULT_CHAT_KB,f,indent=2)
    return DEFAULT_CHAT_KB


def _save_chat_kb(pair):
    kb=_load_chat_kb(); kb.append(pair)
    with open(CHAT_KB_FILE,'w',encoding='utf-8') as f: json.dump(kb,f,indent=2)
    return kb


def _load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE,'r',encoding='utf-8') as f: d=json.load(f)
            if isinstance(d,list): return d
        except Exception: pass
    return []


def _append_chat_history(u,b):
    h=_load_chat_history(); h.append({'timestamp':datetime.utcnow().isoformat()+'Z','user':u,'bot':b})
    with open(CHAT_HISTORY_FILE,'w',encoding='utf-8') as f: json.dump(h,f,indent=2)


CHAT_KB = _load_chat_kb()


def _init_chatbot():
    qs=[i['question'] for i in CHAT_KB]
    v=TfidfVectorizer(ngram_range=(1,2),stop_words='english'); m=v.fit_transform(qs)
    return v,m


chat_vectorizer, chat_q_matrix = _init_chatbot()


def _chatbot_answer(q):
    q=q.strip().lower()
    if not q: return 'Please enter a question.'
    if 'heart' in q and ('symptom' in q or 'risk' in q or 'prevent' in q):
        ans = CHAT_KB[0]['answer']
    else:
        uv=chat_vectorizer.transform([q]); sims=cosine_similarity(uv,chat_q_matrix).flatten()
        bi=int(sims.argmax()); bs=float(sims[bi])
        ans = CHAT_KB[bi]['answer'] if bs>=0.30 else ("I'm here to help with Heart disease, Diabetes, and Parkinson's questions. Please ask in another way or use the relevant form.")
    return ans + ' (Disclaimer: For medical advice, consult a qualified healthcare professional.)'


# ===========================================================================
# ROUTES
# ===========================================================================

@app.route('/')
def index():
    return render_template('index.html')


# ── Heart ──────────────────────────────────────────────────────────────────
@app.route('/heart', methods=['GET','POST'])
def heart():
    prediction_text = None
    firstname = lastname = gender = age = None
    result = None

    if request.method == 'POST':
        try:
            data = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))
        except FileNotFoundError:
            flash("Error: 'heart.csv' was not found.")
            return render_template('heart.html', symptoms=SYMPTOMS_HEART)

        X=data.drop("target",axis=1); y=data["target"]
        rf=RandomForestClassifier(random_state=42); rf.fit(X,y)

        try:
            firstname = request.form.get('firstname')
            lastname  = request.form.get('lastname')
            gender    = request.form.get('gender') or ''
            sex       = 1 if (gender.lower() == 'male') else 0
            age       = float(request.form.get('age'))
            
            # Helper function to get symptom values (form fields are like sym_Chest_pain)
            def get_symptom_value(symptom_label):
                key = f"sym_{symptom_label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}"
                return float(request.form.get(key, 0))
            
            # Get symptom values from form
            symptom_chest_pain = get_symptom_value("Chest pain")
            symptom_breath = get_symptom_value("Shortness of breath")
            symptom_dizzy = get_symptom_value("Dizziness")
            symptom_palp = get_symptom_value("Palpitations")
            symptom_cold = get_symptom_value("Cold Sweats")
            symptom_fatigue = get_symptom_value("Fatigue")
            symptom_jaw = get_symptom_value("Jaw/Arm pain")
            
            # Map physical activity to numeric value
            pa_str = request.form.get('physical_activity', 'Moderate')
            pa_map = {'Low': 0, 'Moderate': 1, 'High': 2}
            physical_activity = pa_map.get(pa_str, 1)
            
            # Map diet quality to numeric value
            diet_str = request.form.get('diet_quality', 'Average')
            diet_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
            diet_quality = diet_map.get(diet_str, 1)
            
            # Create DataFrame with all required columns matching the CSV
            ud = pd.DataFrame({
                "age": [age],
                "sex": [sex],
                "cp": [float(request.form.get('cp'))],
                "trestbps": [float(request.form.get('trestbps'))],
                "chol": [float(request.form.get('chol'))],
                "fbs": [float(request.form.get('fbs'))],
                "restecg": [float(request.form.get('restecg'))],
                "thalach": [float(request.form.get('thalach'))],
                "exang": [float(request.form.get('exang'))],
                "oldpeak": [float(request.form.get('oldpeak'))],
                "slope": [float(request.form.get('slope'))],
                "ca": [float(request.form.get('ca'))],
                "thal": [float(request.form.get('thal'))],
                "symptom_chest_pain_at_rest": [symptom_chest_pain],
                "symptom_shortness_of_breath": [symptom_breath],
                "symptom_dizziness": [symptom_dizzy],
                "symptom_palpitations": [symptom_palp],
                "symptom_cold_sweats": [symptom_cold],
                "symptom_fatigue": [symptom_fatigue],
                "symptom_jaw_arm_pain": [symptom_jaw],
                "smoking": [float(request.form.get('smoking', 0))],
                "alcohol_intake": [float(request.form.get('alcohol', 0))],
                "family_history_heart": [float(request.form.get('family_history_heart', 0))],
                "hypertension_history": [0.0],
                "high_cholesterol_history": [0.0],
                "physical_activity_level": [float(physical_activity)],
                "stress_level": [float(request.form.get('stress_level', 3))],
                "diet_quality": [float(diet_quality)],
                "obesity_history": [0.0],
                "diabetes_history": [0.0]
            })
            # Ensure columns are in the same order as in the original CSV (excluding 'target')
            ud = ud[X.columns]
            pred_int  = int(rf.predict(ud)[0])
            base_prob = float(rf.predict_proba(ud)[0][1])
        except (TypeError, ValueError) as e:
            flash(f"Please enter valid numeric values for all heart fields. ({e})")
            return render_template('heart.html', symptoms=SYMPTOMS_HEART)

        selected_symptoms = get_selected_symptoms(request.form, SYMPTOMS_HEART)
        history = encode_patient_history(request.form)

        result = build_enhanced_result(
            disease="heart", base_prob=base_prob, predicted_positive=(pred_int==1),
            firstname=firstname, lastname=lastname, history=history,
            selected_symptoms=selected_symptoms, family_history_key="family_history_heart",
            subtype_pos="High Cardiovascular Risk", subtype_neg="Low Cardiovascular Risk",
            subtype_desc_pos="Please consult a cardiologist immediately.",
            subtype_desc_neg="No immediate cardiovascular issues detected.",
            recs_pos=["⚠️ Immediate consultation with a cardiologist recommended.",
                      "Monitor blood pressure and cholesterol regularly.",
                      "Avoid smoking, reduce alcohol, and manage stress.",
                      "Consider a cardiac stress test and ECG evaluation."],
            recs_neg=["Maintain a heart-healthy diet (low saturated fat, high fibre).",
                      "Exercise at least 150 minutes per week.",
                      "Schedule annual cardiovascular checkups."],
        )
        prediction_text = result["prediction"]

        try:
            db.session.add(Submission(
                firstname=firstname, lastname=lastname,
                phone_no=request.form.get('phone_no'), area_code=request.form.get('area_code'),
                email=request.form.get('email'), disease='Heart',
                inputs=json.dumps(request.form.to_dict()), result=prediction_text))
            db.session.commit()
        except Exception:
            flash('Warning: could not save record to database.')

    return render_template('heart.html', prediction=prediction_text,
                           firstname=firstname, lastname=lastname, age=age, gender=gender,
                           symptoms=SYMPTOMS_HEART, result=result)


# ── Diabetes ───────────────────────────────────────────────────────────────
@app.route('/diabetes', methods=['GET','POST'])
def diabetes():
    global diabetes_dl_model, diabetes_dl_scaler
    prediction_text = None
    firstname = lastname = gender = age = None
    result = None

    if request.method == 'POST':
        try:
            firstname     = request.form.get('firstname')
            lastname      = request.form.get('lastname')
            gender        = request.form.get('gender')
            gender_val    = 1 if gender.lower()=='male' else 0
            age           = float(request.form.get('age'))
            hypertension  = float(request.form.get('hypertension'))
            heart_disease = float(request.form.get('heart_disease'))
            bmi           = float(request.form.get('bmi'))
            hba1c_level   = float(request.form.get('hba1c_level'))
            glucose_level = float(request.form.get('blood_glucose_level'))
        except (TypeError, ValueError):
            flash("Please enter valid numeric values for all diabetes fields.")
            return render_template('diabetes.html', symptoms=SYMPTOMS_DIABETES)

        if diabetes_dl_model is None or diabetes_dl_scaler is None:
            diabetes_dl_model, diabetes_dl_scaler = _train_diabetes_model()

        if diabetes_dl_model is None or diabetes_dl_scaler is None:
            flash("Error: diabetes model unavailable.")
            return render_template('diabetes.html', symptoms=SYMPTOMS_DIABETES)

        raw_input    = np.array([[gender_val,age,hypertension,heart_disease,bmi,hba1c_level,glucose_level]],dtype=float)
        scaled_input = diabetes_dl_scaler.transform(raw_input)
        base_prob    = float(diabetes_dl_model.predict(scaled_input)[0][0])

        selected_symptoms = get_selected_symptoms(request.form, SYMPTOMS_DIABETES)
        history = encode_patient_history(request.form)

        result = build_enhanced_result(
            disease="diabetes", base_prob=base_prob, predicted_positive=(base_prob>=0.5),
            firstname=firstname, lastname=lastname, history=history,
            selected_symptoms=selected_symptoms, family_history_key="family_history_diabetes",
            subtype_pos="Type 2 Diabetes Indicator", subtype_neg="Healthy Metabolic Profile",
            subtype_desc_pos="High indications of diabetes. Consult an endocrinologist.",
            subtype_desc_neg="Blood sugar parameters look normal.",
            recs_pos=["⚠️ Consult physician for HbA1c and fasting glucose confirmation test.",
                      "Reduce simple carbohydrates and sugary beverages.",
                      "Incorporate at least 30 min of aerobic exercise daily.",
                      "Monitor blood glucose levels at home regularly."],
            recs_neg=["Maintain a balanced diet rich in fibre and whole grains.",
                      "Keep BMI in the 18.5–24.9 normal range.",
                      "Annual HbA1c screening if family history exists."],
        )
        prediction_text = result["prediction"]

        try:
            db.session.add(Submission(
                firstname=firstname, lastname=lastname,
                phone_no=request.form.get('phone_no'), area_code=request.form.get('area_code'),
                email=request.form.get('email'), disease='Diabetes',
                inputs=json.dumps(request.form.to_dict()), result=prediction_text))
            db.session.commit()
        except Exception:
            flash('Warning: could not save record to database.')

    return render_template('diabetes.html', prediction=prediction_text,
                           firstname=firstname, lastname=lastname, age=age, gender=gender,
                           symptoms=SYMPTOMS_DIABETES, result=result)


# ── Parkinson's ────────────────────────────────────────────────────────────
@app.route('/parkinson', methods=['GET','POST'])
def parkinson():
    global parkinson_dl_model, parkinson_dl_scaler
    prediction_text = None
    firstname = lastname = gender = age = None
    result = None

    if request.method == 'POST':
        try:
            firstname           = request.form.get('firstname')
            lastname            = request.form.get('lastname')
            gender              = request.form.get('gender')
            age                 = float(request.form.get('age'))
            mdvp_fo             = float(request.form.get('MDVP:Fo(Hz)'))
            mdvp_fhi            = float(request.form.get('MDVP:Fhi(Hz)'))
            mdvp_flo            = float(request.form.get('MDVP:Flo(Hz)'))
            mdvp_jitter_percent = float(request.form.get('MDVP:Jitter(%)'))
            mdvp_jitter_abs     = float(request.form.get('MDVP:Jitter(Abs)'))
            mdvp_rap            = float(request.form.get('MDVP:RAP'))
            mdvp_ppq            = float(request.form.get('MDVP:PPQ'))
            jitter_ddp          = float(request.form.get('Jitter:DDP'))
            mdvp_shimmer        = float(request.form.get('MDVP:Shimmer'))
            mdvp_shimmer_db     = float(request.form.get('MDVP:Shimmer(dB)'))
            shimmer_apq3        = float(request.form.get('Shimmer:APQ3'))
            shimmer_apq5        = float(request.form.get('Shimmer:APQ5'))
            mdvp_apq            = float(request.form.get('MDVP:APQ'))
            shimmer_dda         = float(request.form.get('Shimmer:DDA'))
            nhr                 = float(request.form.get('NHR'))
            hnr                 = float(request.form.get('HNR'))
            rpde                = float(request.form.get('RPDE'))
            dfa                 = float(request.form.get('DFA'))
            spread1             = float(request.form.get('spread1'))
            spread2             = float(request.form.get('spread2'))
            d2                  = float(request.form.get('D2'))
            ppe                 = float(request.form.get('PPE'))
        except (TypeError, ValueError):
            flash("Please enter valid numeric values for all Parkinson's fields.")
            return render_template('parkinson.html', symptoms=SYMPTOMS_PARKINSON)

        if parkinson_dl_model is None or parkinson_dl_scaler is None:
            parkinson_dl_model, parkinson_dl_scaler = _train_parkinson_model()

        if parkinson_dl_model is None or parkinson_dl_scaler is None:
            flash("Error: Parkinson model unavailable.")
            return render_template('parkinson.html', symptoms=SYMPTOMS_PARKINSON)

        raw_input = np.array([[mdvp_fo,mdvp_fhi,mdvp_flo,mdvp_jitter_percent,mdvp_jitter_abs,
            mdvp_rap,mdvp_ppq,jitter_ddp,mdvp_shimmer,mdvp_shimmer_db,shimmer_apq3,shimmer_apq5,
            mdvp_apq,shimmer_dda,nhr,hnr,rpde,dfa,spread1,spread2,d2,ppe]],dtype=float)
        scaled_input = parkinson_dl_scaler.transform(raw_input)
        base_prob    = float(parkinson_dl_model.predict(scaled_input)[0][0])

        selected_symptoms = get_selected_symptoms(request.form, SYMPTOMS_PARKINSON)
        history = encode_patient_history(request.form)

        result = build_enhanced_result(
            disease="parkinson", base_prob=base_prob, predicted_positive=(base_prob>=0.5),
            firstname=firstname, lastname=lastname, history=history,
            selected_symptoms=selected_symptoms, family_history_key="family_history_parkinson",
            subtype_pos="Motor Symptom Risk Detected", subtype_neg="No Significant Neurodegenerative Risk",
            subtype_desc_pos="Voice acoustics suggest neurodegenerative patterns.",
            subtype_desc_neg="Vocal features are within normal range.",
            recs_pos=["⚠️ See a neurologist for a clinical Parkinson's evaluation.",
                      "Document onset and frequency of any tremors or stiffness.",
                      "Consider DaT-SPECT imaging if clinically indicated.",
                      "Engage in regular physical therapy and exercise."],
            recs_neg=["Maintain an active lifestyle with regular aerobic exercise.",
                      "Monitor any new tremors, gait changes, or stiffness.",
                      "Regular checkups if age > 60 or family history present."],
        )
        prediction_text = result["prediction"]

        try:
            db.session.add(Submission(
                firstname=firstname, lastname=lastname,
                phone_no=request.form.get('phone_no'), area_code=request.form.get('area_code'),
                email=request.form.get('email'), disease='Parkinson',
                inputs=json.dumps(request.form.to_dict()), result=prediction_text))
            db.session.commit()
        except Exception:
            flash('Warning: could not save record to database.')

    return render_template('parkinson.html', prediction=prediction_text,
                           firstname=firstname, lastname=lastname, age=age, gender=gender,
                           symptoms=SYMPTOMS_PARKINSON, result=result)


# ── PDF Report Download ────────────────────────────────────────────────────
@app.route('/report/download', methods=['POST'])
def download_report():
    data = request.get_json(force=True, silent=True) or {}
    firstname    = data.get('firstname','Patient')
    lastname     = data.get('lastname','')
    disease      = data.get('disease','Unknown')
    prediction   = data.get('prediction','N/A')
    probability  = data.get('probability',0)
    risk_level   = data.get('risk_level','N/A')
    matched_syms = data.get('matched_symptoms',[])
    explanation  = data.get('explanation',[])
    recs         = data.get('recommendations',[])
    timestamp    = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    rc = {"High":"#dc2626","Medium":"#d97706","Low":"#16a34a"}.get(risk_level,"#64748b")

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>HealthCure Report – {firstname} {lastname}</title>
<style>
body{{font-family:Arial,sans-serif;color:#1a2332;margin:40px;font-size:13px;}}
h1{{color:#2E86DE;font-size:22px;margin-bottom:4px;}}
.subtitle{{color:#64748b;font-size:12px;margin-bottom:24px;}}
.section{{margin-bottom:20px;}}
.st{{font-weight:bold;font-size:14px;border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-bottom:12px;}}
.kv{{display:flex;gap:40px;flex-wrap:wrap;}}
.kv-item{{margin-bottom:10px;}}
.kv-label{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;}}
.kv-value{{font-size:15px;font-weight:bold;}}
.rb{{display:inline-block;padding:3px 12px;border-radius:12px;font-weight:bold;font-size:12px;color:{rc};background:{rc}22;border:1px solid {rc}55;}}
ul{{padding-left:18px;line-height:1.9;}}
.sym{{display:inline-block;background:#dbeafe;color:#1d4ed8;border-radius:8px;padding:2px 10px;font-size:11px;margin:2px;}}
.exp{{padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:12px;color:#334155;}}
.footer{{margin-top:32px;padding-top:12px;border-top:1px solid #e2e8f0;font-size:10px;color:#94a3b8;}}
.disc{{background:#fef9c3;border:1px solid #fef08a;border-radius:6px;padding:10px 14px;font-size:11px;color:#854d0e;margin-top:16px;}}
</style></head><body>
<h1>🏥 HealthCure Patient Report</h1>
<div class="subtitle">Generated: {timestamp} | Report ID: HC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}</div>
<div class="section"><div class="st">Patient Information</div>
<div class="kv">
<div class="kv-item"><div class="kv-label">Full Name</div><div class="kv-value">{firstname} {lastname}</div></div>
<div class="kv-item"><div class="kv-label">Disease Assessed</div><div class="kv-value">{disease}</div></div>
</div></div>
<div class="section"><div class="st">Prediction Result</div>
<div class="kv">
<div class="kv-item"><div class="kv-label">Diagnosis</div><div class="kv-value">{prediction}</div></div>
<div class="kv-item"><div class="kv-label">Enhanced Confidence</div><div class="kv-value">{probability}%</div></div>
<div class="kv-item"><div class="kv-label">Risk Level</div><div class="kv-value"><span class="rb">{risk_level} Risk</span></div></div>
</div></div>
{'<div class="section"><div class="st">Matched Symptoms</div>'+''.join(f'<span class="sym">{s}</span>' for s in matched_syms)+'</div>' if matched_syms else ''}
<div class="section"><div class="st">AI Confidence Explanation</div>
{''.join(f'<div class="exp">{e}</div>' for e in explanation)}
</div>
<div class="section"><div class="st">Recommendations</div>
<ul>{''.join(f'<li>{r}</li>' for r in recs)}</ul>
</div>
<div class="disc">⚠️ <strong>Medical Disclaimer:</strong> This report is for educational purposes only. It is NOT a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare professional.</div>
<div class="footer">HealthCure AI Multi-Disease Prediction System | ANN & Random Forest Models</div>
</body></html>"""

    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    resp.headers['Content-Disposition'] = f'attachment; filename="HealthCure_{firstname}_{lastname}.html"'
    return resp


# ── Records ────────────────────────────────────────────────────────────────
@app.route('/records')
def records():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login', next=request.path))
    disease = request.args.get('disease')
    try:
        if disease:
            submissions = Submission.query.filter_by(disease=disease).order_by(Submission.timestamp.desc()).all()
        else:
            submissions = Submission.query.order_by(Submission.timestamp.desc()).all()
    except Exception:
        submissions = []
    return render_template('records.html', submissions=submissions, json=json, disease=disease)


@app.route('/records/edit/<int:record_id>', methods=['GET','POST'])
def edit_record(record_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login', next=url_for('records')))
    record = Submission.query.get_or_404(record_id)
    if request.method == 'POST':
        record.firstname = request.form.get('firstname')
        record.lastname  = request.form.get('lastname')
        record.phone_no  = request.form.get('phone_no')
        record.area_code = request.form.get('area_code')
        record.email     = request.form.get('email')
        record.disease   = request.form.get('disease')
        record.result    = request.form.get('result')
        try: record.inputs = json.dumps(json.loads(request.form.get('inputs') or '{}'))
        except Exception:
            flash('Invalid JSON for inputs.','danger')
            return render_template('edit_record.html', record=record)
        try:
            db.session.commit(); flash('Record updated successfully.','success')
            return redirect(url_for('records'))
        except Exception:
            db.session.rollback(); flash('Error saving record update.','danger')
    return render_template('edit_record.html', record=record)


@app.route('/records/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login', next=url_for('records')))
    record = Submission.query.get_or_404(record_id)
    try:
        db.session.delete(record); db.session.commit(); flash('Record deleted successfully.','success')
    except Exception:
        db.session.rollback(); flash('Could not delete record.','danger')
    return redirect(url_for('records'))


# ── Chat ───────────────────────────────────────────────────────────────────
@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/chat/query', methods=['POST'])
def chat_query():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get('question') or '').strip()
    if not q:
        return {'answer': 'Please enter a question.'}, 400
    ans = _chatbot_answer(q)
    _append_chat_history(q, ans)
    return {'answer': ans}


@app.route('/chat/history', methods=['GET'])
def chat_history():
    return {'history': list(reversed(_load_chat_history()))}


@app.route('/chat/history/clear', methods=['POST'])
def chat_history_clear():
    with open(CHAT_HISTORY_FILE,'w',encoding='utf-8') as f: json.dump([],f,indent=2)
    return {'status':'cleared'}


@app.route('/chat/kb/add', methods=['POST'])
def chat_kb_add():
    data = request.get_json(force=True,silent=True) or {}
    q=(data.get('question') or '').strip().lower(); a=(data.get('answer') or '').strip()
    if not q or not a: return {'error':'question and answer required'},400
    _save_chat_kb({'question':q,'answer':a})
    global chat_vectorizer, chat_q_matrix, CHAT_KB
    CHAT_KB=_load_chat_kb(); chat_vectorizer,chat_q_matrix=_init_chatbot()
    return {'status':'added'}


# ── Login/Logout ───────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    next_url = request.args.get('next') or url_for('records')
    if request.method == 'POST':
        if request.form.get('admin_id')==app.config['ADMIN_ID'] and request.form.get('admin_pwd')==app.config['ADMIN_PWD']:
            session['admin_logged_in']=True; flash('Logged in successfully.','success')
            return redirect(request.form.get('next') or url_for('records'))
        flash('Invalid credentials. Please try again.','danger')
    return render_template('login.html', next=next_url)


@app.route('/logout')
def logout():
    session.pop('admin_logged_in',None); flash('Logged out.','info')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

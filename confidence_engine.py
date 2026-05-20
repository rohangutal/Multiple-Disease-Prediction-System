"""
confidence_engine.py
---------------------
Intelligent confidence-score enhancement layer for the Multi-Disease Prediction System.

Formula (from project spec):
    Final Confidence = Base_Model_Prob
                     + (Symptom_Match_Score  * W1)
                     + (History_Impact_Score * W2)
                     [adjusted by rule-based layer]

All inputs and outputs are in probability space [0.0 – 1.0].
The final score is also exposed as 0–100 percentage.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Weights (tunable)
# ---------------------------------------------------------------------------
W1_SYMPTOM  = 0.15   # max symptom contribution: 15pp
W2_HISTORY  = 0.08   # max history contribution:  8pp

# Rule-based adjustment deltas
RULE_STRONG_MATCH_BOOST    =  0.08   # model positive + strong symptoms → boost
RULE_MISMATCH_PENALTY      = -0.06   # model positive + symptom mismatch → reduce
RULE_BORDERLINE_BOOST      =  0.05   # model borderline + strong symptoms → boost
BORDERLINE_THRESHOLD       =  0.42   # prob below this is "borderline"
STRONG_MATCH_THRESHOLD     =  0.60   # symptom_match_score above this is "strong"

# ---------------------------------------------------------------------------
# History impact scoring
# ---------------------------------------------------------------------------

def _compute_history_impact(history: dict, family_history_key: str | None) -> float:
    """
    Convert patient history dict into a [0, 1] impact score.
    Reflects lifestyle risk factors that increase disease probability.
    """
    score = 0.0
    total = 0.0

    # Smoking: 0/1
    score += history.get("smoking", 0) * 0.30
    total += 0.30

    # Alcohol: 0/1
    score += history.get("alcohol", 0) * 0.15
    total += 0.15

    # Family history (disease-specific key): 0/1
    fam = history.get(family_history_key, 0) if family_history_key else 0
    score += fam * 0.25
    total += 0.25

    # Stress level 1–5; above 3 is elevated
    stress = int(history.get("stress_level", 3))
    stress_contribution = max(0, stress - 3) / 2.0   # 0, 0.5, or 1.0
    score += stress_contribution * 0.15
    total += 0.15

    # Physical activity: Low=0, Moderate=1, High=2 → invert
    pa = history.get("physical_activity", 1)
    pa_risk = (2 - pa) / 2.0   # Low=1.0 risk, High=0.0 risk
    score += pa_risk * 0.10
    total += 0.10

    # Diet quality: Poor=0, Average=1, Good=2, Excellent=3 → invert
    dq = history.get("diet_quality", 1)
    dq_risk = (3 - dq) / 3.0   # Poor=1.0 risk, Excellent=0.0 risk
    score += dq_risk * 0.05
    total += 0.05

    return score / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Rule-based adjustment layer
# ---------------------------------------------------------------------------

def _apply_rules(base_prob: float,
                 predicted_positive: bool,
                 symptom_match_score: float,
                 mismatch_flag: bool) -> float:
    """
    Add a rule-based delta AFTER the formula adjustment.

    Rules (from project spec):
      IF positive AND strong match  → significant boost
      IF positive AND mismatch      → reduce confidence
      IF borderline AND strong match → moderate boost
    """
    delta = 0.0

    if predicted_positive and symptom_match_score >= STRONG_MATCH_THRESHOLD:
        delta += RULE_STRONG_MATCH_BOOST

    if predicted_positive and mismatch_flag:
        delta += RULE_MISMATCH_PENALTY

    if (not predicted_positive or base_prob < BORDERLINE_THRESHOLD) \
            and symptom_match_score >= STRONG_MATCH_THRESHOLD:
        delta += RULE_BORDERLINE_BOOST

    return delta


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def enhance_confidence(
    base_prob: float,
    predicted_positive: bool,
    symptom_result: dict,
    history: dict,
    family_history_key: str | None = None,
) -> dict:
    """
    Compute the enhanced confidence score and risk metadata.

    Parameters
    ----------
    base_prob           : raw model output probability [0–1]
    predicted_positive  : True if model predicts disease
    symptom_result      : output dict from symptom_engine.score_symptoms()
    history             : encoded patient history dict (from app.encode_patient_history)
    family_history_key  : history dict key for the relevant family history flag

    Returns
    -------
    dict with keys:
        final_prob          float 0–1
        prob_percent        float 0–100
        risk_level          str  'Low' | 'Medium' | 'High'
        base_prob           float
        symptom_adjustment  float (contribution of symptoms)
        history_adjustment  float (contribution of history)
        rule_adjustment     float (contribution of rule layer)
        history_impact_score float 0–1
        explanation         list[str]  human-readable explanation items
        confidence_tier     str 'Very High' | 'High' | 'Moderate' | 'Low'
    """
    base_prob = max(0.0, min(1.0, base_prob))

    match_score   = symptom_result.get("symptom_match_score", 0.0)
    mismatch_flag = symptom_result.get("mismatch_flag", False)
    n_symptoms    = symptom_result.get("raw_symptom_count", 0)

    # ── Formula ──────────────────────────────────────────────────────────
    history_impact = _compute_history_impact(history, family_history_key)

    symptom_adj = match_score * W1_SYMPTOM
    history_adj = history_impact * W2_HISTORY

    intermediate = base_prob + symptom_adj + history_adj

    # ── Rule-based layer ─────────────────────────────────────────────────
    rule_adj = _apply_rules(base_prob, predicted_positive, match_score, mismatch_flag)

    final_prob = max(0.01, min(0.99, intermediate + rule_adj))
    prob_percent = round(final_prob * 100, 1)

    # ── Risk classification ───────────────────────────────────────────────
    if prob_percent >= 65:
        risk_level = "High"
    elif prob_percent >= 35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # ── Confidence tier (model certainty) ────────────────────────────────
    distance = abs(final_prob - 0.5)
    if distance >= 0.35:
        confidence_tier = "Very High"
    elif distance >= 0.22:
        confidence_tier = "High"
    elif distance >= 0.10:
        confidence_tier = "Moderate"
    else:
        confidence_tier = "Low"

    # ── Human-readable explanation ────────────────────────────────────────
    explanation = _build_explanation(
        base_prob, symptom_adj, history_adj, rule_adj,
        match_score, mismatch_flag, history_impact,
        n_symptoms, symptom_result.get("matched_symptoms", []),
        predicted_positive, final_prob
    )

    return {
        "final_prob":           round(final_prob, 4),
        "prob_percent":         prob_percent,
        "risk_level":           risk_level,
        "base_prob":            round(base_prob, 4),
        "symptom_adjustment":   round(symptom_adj, 4),
        "history_adjustment":   round(history_adj, 4),
        "rule_adjustment":      round(rule_adj, 4),
        "history_impact_score": round(history_impact, 4),
        "explanation":          explanation,
        "confidence_tier":      confidence_tier,
    }


def _build_explanation(base_prob, sym_adj, hist_adj, rule_adj,
                        match_score, mismatch, history_impact,
                        n_symptoms, matched_names,
                        predicted_positive, final_prob):
    """Generate a list of explanation strings for the UI."""
    lines = []

    # Base model
    bp_pct = round(base_prob * 100, 1)
    lines.append(f"📊 Base ML model probability: {bp_pct}%")

    # Symptoms
    if n_symptoms > 0:
        sym_pct = round(sym_adj * 100, 1)
        lines.append(
            f"🔍 {n_symptoms} symptom(s) reported with {round(match_score*100)}% "
            f"profile match → +{sym_pct}% adjustment"
        )
        if matched_names:
            joined = ", ".join(matched_names[:4])
            suffix = f" (+{len(matched_names)-4} more)" if len(matched_names) > 4 else ""
            lines.append(f"   Matched: {joined}{suffix}")
    else:
        lines.append("🔍 No symptoms reported — symptom contribution: 0%")

    # History
    if history_impact > 0.05:
        hist_pct = round(hist_adj * 100, 1)
        lines.append(f"🏥 Lifestyle & history risk factor score: {round(history_impact*100)}% → +{hist_pct}% adjustment")

    # Rule-based
    if rule_adj > 0:
        lines.append(f"⚡ Rule-based boost applied (strong symptom-prediction alignment): +{round(rule_adj*100, 1)}%")
    elif rule_adj < 0:
        lines.append(f"⚠️ Rule-based penalty applied (symptom-prediction mismatch): {round(rule_adj*100, 1)}%")

    # Mismatch note
    if mismatch:
        lines.append("⚠️ Note: selected symptoms do not strongly align with the predicted disease — interpret with caution.")

    # Final
    lines.append(f"✅ Final enhanced confidence score: {round(final_prob*100, 1)}%")

    return lines

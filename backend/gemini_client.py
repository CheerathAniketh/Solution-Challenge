import json
import os
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PERSONAS = {
    "student":  "a student learning about AI fairness for the first time",
    "ngo":      "an NGO worker with no technical background who needs to act on this",
    "policy":   "a policy maker concerned about legal and ethical risk",
    "general":  "a general audience with no technical background",
}


def _is_api_error(e: Exception) -> bool:
    err = str(e)
    return any(code in err for code in [
        "429", "RESOURCE_EXHAUSTED",
        "400", "INVALID_ARGUMENT",
        "API_KEY_INVALID", "API Key not found"
    ])


def _fallback_explanation(bias_report) -> str:
    di = bias_report.get("di", 0)
    spd = bias_report.get("spd", 0)
    severity = bias_report.get("severity", "unknown")
    groups = bias_report.get("group_stats", {})

    group_lines = ""
    if groups:
        sorted_groups = sorted(groups.items(), key=lambda x: x[1]["positive_rate"])
        least = sorted_groups[0]
        most = sorted_groups[-1]
        group_lines = (
            f"The most favored group is '{most[0]}' with an approval rate of "
            f"{most[1]['positive_rate']*100:.1f}%, while '{least[0]}' has only "
            f"{least[1]['positive_rate']*100:.1f}%."
        )

    bias_line = (
        "This level of disparity means equally qualified people are being treated differently "
        "based on a protected attribute. In a hiring context, this could mean qualified candidates "
        "are being rejected due to gender, race, or other protected characteristics. "
        if di < 0.8 else
        "The model appears to be treating groups fairly based on these metrics. "
    )

    return (
        f"This dataset shows {'significant' if di < 0.8 else 'no significant'} bias. "
        f"The Disparate Impact Ratio is {di:.3f} "
        f"({'below' if di < 0.8 else 'above'} the legal threshold of 0.80), "
        f"and the outcome gap between groups is {spd*100:.1f} percentage points. "
        f"Severity: {severity.upper()}. {group_lines}\n\n"
        f"{bias_line}\n\n"
        f"To address this: first, review and remove any proxy features that indirectly encode "
        f"the sensitive attribute. Second, retrain the model on a rebalanced dataset where all "
        f"groups have equal representation in positive outcomes. *"
    )


def _fallback_fixes(bias_report, shap_data) -> list:
    features = shap_data.get("top_features", [])
    proxy_keywords = ["gap", "zip", "cost", "insurance", "prestige", "address"]
    proxies = [f["feature"] for f in features if any(p in f["feature"].lower() for p in proxy_keywords)]

    fixes = []
    if proxies:
        fixes.append(f"Remove or replace proxy features: {', '.join(proxies)} — these correlate with the sensitive attribute and cause indirect discrimination.")
    fixes.append("Rebalance your training dataset so all demographic groups have equal representation in positive outcomes.")
    fixes.append("Apply post-processing threshold calibration to equalize false positive and false negative rates across groups.")
    return fixes[:3]


def _parse_json(text: str):
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return json.loads(cleaned)


def explain_results(bias_report, shap_data, audience="ngo"):
    persona = PERSONAS.get(audience, PERSONAS["ngo"])

    prompt = f"""
You are an AI fairness expert. Be extremely concise.

Bias analysis results:
- Groups: {bias_report['group_stats']}
- Disparate Impact: {bias_report['di']} (below 0.8 = biased)
- Outcome gap: {bias_report['spd']} (above 0.1 = biased)
- Severity: {bias_report['severity']}
- Top features driving decisions: {shap_data['top_features']}

Write exactly 3 sentences. No more.
Sentence 1: Which group is disadvantaged and by how much (use the actual numbers).
Sentence 2: Which feature(s) are causing it and why that's a problem.
Sentence 3: One specific fix.

Rules:
- No intros, no conclusions, no filler
- Use plain language suited for {persona}
- Name the actual columns, not generic terms
- Total response must be under 60 words
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        if _is_api_error(e):
            return _fallback_explanation(bias_report)
        raise e


def suggest_fixes(bias_report, shap_data):
    prompt = f"""
Given this bias analysis: {bias_report}
And these influential features: {shap_data['top_features']}

Suggest exactly 3 specific actionable fixes.
Return a JSON array only. No markdown, no backticks, no explanation.
Example format: ["fix 1", "fix 2", "fix 3"]
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        try:
            return _parse_json(response.text)
        except (ValueError, json.JSONDecodeError):
            return _fallback_fixes(bias_report, shap_data)
    except Exception as e:
        if _is_api_error(e):
            return _fallback_fixes(bias_report, shap_data)
        raise e


def explain_whatif(original, modified, delta, dropped_features, audience="ngo"):
    """
    Explain what changed in bias metrics after dropping specific features.
    Returns a plain-language summary of the before/after comparison.
    """
    persona = PERSONAS.get(audience, PERSONAS["ngo"])
    dropped_str = ", ".join(dropped_features)

    improved = []
    worsened = []

    if delta["spd"] > 0.01:
        improved.append(f"outcome gap narrowed by {abs(delta['spd']):.3f} points")
    elif delta["spd"] < -0.01:
        worsened.append(f"outcome gap widened by {abs(delta['spd']):.3f} points")

    if delta["di"] > 0.01:
        improved.append(f"Disparate Impact improved from {original['di']:.3f} to {modified['di']:.3f}")
    elif delta["di"] < -0.01:
        worsened.append(f"Disparate Impact worsened from {original['di']:.3f} to {modified['di']:.3f}")

    if delta["eod"] > 0.01:
        improved.append(f"error rate gap between groups dropped by {abs(delta['eod']):.3f}")
    elif delta["eod"] < -0.01:
        worsened.append(f"error rate gap between groups grew by {abs(delta['eod']):.3f}")

    di_threshold_crossed = original["di"] < 0.8 and modified["di"] >= 0.8

    prompt = f"""Be extremely concise. 3 sentences max.

Features dropped: {dropped_str}
Audience: {persona}

Before → After:
- Disparate Impact: {original['di']:.3f} → {modified['di']:.3f} (need ≥ 0.8)
- SPD: {original['spd']:.3f} → {modified['spd']:.3f} (need ≤ 0.1)
- Equalized Odds: {original['eod']:.3f} → {modified['eod']:.3f} (need ≤ 0.1)
- Threshold crossed: {di_threshold_crossed}

Sentence 1: What improved and what got worse, using the actual numbers.
Sentence 2: Is the model fair now? Yes or no, with the DI number.
Sentence 3: One specific next step.

Hard limit: under 50 words total. No filler, no intros.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        if _is_api_error(e):
            return _fallback_whatif_explanation_gemini(original, modified, delta, dropped_features)
        raise e


def _fallback_whatif_explanation_gemini(original, modified, delta, dropped):
    """Fallback when Gemini is unavailable for whatif explanations."""
    dropped_str = ", ".join(dropped)
    spd_dir = "decreased" if delta["spd"] > 0 else "increased"
    di_dir  = "improved"  if delta["di"]  > 0 else "worsened"
    eod_dir = "decreased" if delta["eod"] > 0 else "increased"

    threshold_note = ""
    if original["di"] < 0.8 and modified["di"] >= 0.8:
        threshold_note = " The model has now crossed the legal fairness threshold of 0.80 — this is a meaningful improvement."
    elif modified["di"] < 0.8:
        threshold_note = f" The Disparate Impact is still below the legal threshold of 0.80 (currently {modified['di']:.3f}), so further action is needed."

    return (
        f"After removing {dropped_str}, the outcome gap between groups {spd_dir} "
        f"by {abs(delta['spd']):.3f} points, the Disparate Impact ratio {di_dir} "
        f"from {original['di']:.3f} to {modified['di']:.3f}, and the error rate gap "
        f"{eod_dir} from {original['eod']:.3f} to {modified['eod']:.3f}.{threshold_note}\n\n"
        f"{'These features appear to have been contributing to bias in the model — consider permanently removing them from your dataset.' if delta['di'] > 0 else 'Removing these features did not reduce bias. The discrimination likely comes from other features — try dropping different columns or rebalancing the dataset.'} *"
    )
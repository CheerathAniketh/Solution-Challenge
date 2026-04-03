from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from typing import List
from fastapi.responses import Response
from pdf_exporter import generate_audit_pdf

from analyzer import analyze_bias, compute_intersectionality, compute_eod
from trainer import train_and_evaluate
from explainer import get_shap_values
from gemini_client import explain_results, suggest_fixes, explain_whatif

app = FastAPI(title="EquiLens AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session cache ───────────────────────────────────────────────────
# Stores the last uploaded dataset so /whatif doesn't require re-uploading.
# Single-user / demo scope — fine for GDG judging context.
session_cache: dict = {}


# ─── Pydantic models ──────────────────────────────────────────────────────────

class WhatIfRequest(BaseModel):
    drop_features: List[str]
    audience: str = "ngo"


# ─── /analyze ─────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_col: str = "",
    sensitive_col: str = "",
    sensitive_col_2: str = "",
    audience: str = "ngo",
):
    df = pd.read_csv(file.file)

    # Cache the raw dataframe and column selections for /whatif
    session_cache["df"] = df
    session_cache["target_col"] = target_col
    session_cache["sensitive_col"] = sensitive_col

    stats = analyze_bias(df, target_col, sensitive_col)
    model, X_train, X_test, y_test, curves, y_pred, y_prob, sensitive_test = train_and_evaluate(
        df, target_col, sensitive_col
    )
    shap_data = get_shap_values(model, X_train, X_test)

    eod_data = compute_eod(y_test, y_pred, sensitive_test)
    stats["eod"] = eod_data["eod"]
    stats["eod_details"] = eod_data

    intersectionality = None
    if sensitive_col_2 and sensitive_col_2 != sensitive_col and sensitive_col_2 in df.columns:
        intersectionality = compute_intersectionality(df, target_col, sensitive_col, sensitive_col_2)

    try:
        explanation = explain_results(stats, shap_data, audience=audience)
    except Exception as e:
        explanation = f"Gemini unavailable: {str(e)}"

    try:
        fixes = suggest_fixes(stats, shap_data)
    except Exception as e:
        fixes = [
            "Rebalance your dataset so all groups have equal representation.",
            "Remove proxy features that correlate with the sensitive attribute.",
            "Collect more representative data from underrepresented groups.",
        ]

    return {
        "stats": stats,
        "shap": shap_data,
        "explanation": explanation,
        "fixes": fixes,
        "curves": curves,
        "intersectionality": intersectionality,
    }


# ─── /whatif/features ─────────────────────────────────────────────────────────
# Defined BEFORE /whatif POST to avoid FastAPI route shadowing.

@app.get("/whatif/features")
async def whatif_features():
    if "df" not in session_cache:
        raise HTTPException(
            status_code=400,
            detail="No dataset in cache. Run /analyze first."
        )
    df = session_cache["df"]
    target_col = session_cache["target_col"]
    sensitive_col = session_cache["sensitive_col"]
    features = [
        c for c in df.columns
        if c not in [target_col, sensitive_col]
    ]
    return {"features": features}


# ─── /whatif ──────────────────────────────────────────────────────────────────

@app.post("/whatif")
async def whatif(body: WhatIfRequest):
    if "df" not in session_cache:
        raise HTTPException(
            status_code=400,
            detail="No dataset in cache. Run /analyze first to upload a CSV."
        )

    df = session_cache["df"]
    target_col = session_cache["target_col"]
    sensitive_col = session_cache["sensitive_col"]

    # Validate requested features exist
    all_features = [c for c in df.columns if c != target_col]
    invalid = [f for f in body.drop_features if f not in all_features]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Features not found in dataset: {invalid}. Available: {all_features}"
        )

    # ── Original metrics — raw label approval rates + model EOD ───────────────
    original_stats = analyze_bias(df, target_col, sensitive_col)
    _, X_train_orig, X_test_orig, y_test_orig, _, y_pred_orig, _, sensitive_test_orig = train_and_evaluate(
        df, target_col, sensitive_col
    )
    eod_orig = compute_eod(y_test_orig, y_pred_orig, sensitive_test_orig)
    original_stats["eod"] = eod_orig["eod"]

    # ── Modified: retrain on reduced feature set ───────────────────────────────
    df_modified = df.drop(columns=body.drop_features, errors="ignore")

    model_mod, X_train_mod, X_test_mod, y_test_mod, _, y_pred_mod, _, sensitive_test_mod = train_and_evaluate(
        df_modified, target_col, sensitive_col
    )

    # Compute SPD/DI from MODEL PREDICTIONS (not raw labels).
    # Dropping features only affects what the model decides, not the ground-truth
    # labels in the CSV — so we must measure the model's output per group.
    pred_df = pd.DataFrame({
        sensitive_col: sensitive_test_mod.values,
        target_col: y_pred_mod,
    })
    modified_stats = analyze_bias(pred_df, target_col, sensitive_col)

    eod_mod = compute_eod(y_test_mod, y_pred_mod, sensitive_test_mod)
    modified_stats["eod"] = eod_mod["eod"]

    shap_modified = get_shap_values(model_mod, X_train_mod, X_test_mod)

    # ── Delta — positive always means improvement ──────────────────────────────
    delta = {
        "spd": round(original_stats["spd"] - modified_stats["spd"], 4),  # lower is better
        "di":  round(modified_stats["di"]  - original_stats["di"],  4),  # higher is better
        "eod": round(original_stats["eod"] - modified_stats["eod"], 4),  # lower is better
    }

    # ── Gemini explanation ─────────────────────────────────────────────────────
    try:
        whatif_explanation = explain_whatif(
            original=original_stats,
            modified=modified_stats,
            delta=delta,
            dropped_features=body.drop_features,
            audience=body.audience,
        )
    except Exception:
        whatif_explanation = _fallback_whatif_explanation(
            original_stats, modified_stats, delta, body.drop_features
        )

    return {
        "original": {
            "spd":         original_stats["spd"],
            "di":          original_stats["di"],
            "eod":         original_stats["eod"],
            "severity":    original_stats["severity"],
            "group_stats": original_stats["group_stats"],
        },
        "modified": {
            "spd":         modified_stats["spd"],
            "di":          modified_stats["di"],
            "eod":         modified_stats["eod"],
            "severity":    modified_stats["severity"],
            "group_stats": modified_stats["group_stats"],
        },
        "delta":              delta,
        "dropped_features":   body.drop_features,
        "remaining_features": [
            c for c in df_modified.columns
            if c not in [target_col, sensitive_col]
        ],
        "shap":        shap_modified,
        "explanation": whatif_explanation,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fallback_whatif_explanation(original, modified, delta, dropped):
    spd_dir = "decreased" if delta["spd"] > 0 else "increased"
    di_dir  = "improved"  if delta["di"]  > 0 else "worsened"
    eod_dir = "decreased" if delta["eod"] > 0 else "increased"
    dropped_str = ", ".join(dropped)

    threshold_note = ""
    if original["di"] < 0.8 and modified["di"] >= 0.8:
        threshold_note = " The model has now crossed the legal fairness threshold of 0.80 — this is a meaningful improvement."
    elif modified["di"] < 0.8:
        threshold_note = (
            f" The Disparate Impact is still below the legal threshold of 0.80 "
            f"(currently {modified['di']:.3f}), so further action is needed."
        )

    outcome = (
        "These features appear to have been contributing to bias — consider permanently removing them."
        if delta["di"] > 0
        else "Removing these features did not reduce bias. The discrimination likely comes from other features — try dropping different columns or rebalancing the dataset."
    )

    return (
        f"After dropping {dropped_str}, the outcome gap between groups {spd_dir} "
        f"by {abs(delta['spd']):.3f} points. "
        f"The Disparate Impact ratio {di_dir} from {original['di']:.3f} to {modified['di']:.3f}."
        f"{threshold_note} "
        f"Equalized Odds {eod_dir} from {original['eod']:.3f} to {modified['eod']:.3f}.\n\n"
        f"{outcome} *"
    )

@app.post("/export-pdf")
async def export_pdf(payload: dict):
    pdf_bytes = generate_audit_pdf(payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=equilens_audit.pdf"},
    )


# ─── Static frontend — mount LAST, after all API routes ───────────────────────
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
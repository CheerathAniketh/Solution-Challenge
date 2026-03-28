from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

from analyzer import analyze_bias, compute_intersectionality
from trainer import train_and_evaluate
from explainer import get_shap_values
from gemini_client import explain_results, suggest_fixes

app = FastAPI(title="EquiLens AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_col: str = "",
    sensitive_col: str = "",
    sensitive_col_2: str = "",
    audience: str = "ngo",
):
    df = pd.read_csv(file.file)

    stats = analyze_bias(df, target_col, sensitive_col)
    model, X_train, X_test, y_test, curves = train_and_evaluate(df, target_col, sensitive_col)
    shap_data = get_shap_values(model, X_train, X_test)

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


# Mount LAST — after all API routes
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
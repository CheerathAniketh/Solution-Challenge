# EquiLens AI
### AI-powered bias detection for non-technical users

> "Amazon's hiring AI downgraded women's CVs. COMPAS
> flagged Black defendants at 2× the rate. These failures
> could have been caught. EquiLens catches them."

## The Problem
AI makes life-changing decisions about jobs, loans, and
healthcare. When trained on biased historical data, these
systems don't just repeat discrimination — they amplify
it at scale, silently, with no accountability.

## The Solution
EquiLens gives any organization — NGO, school, small
business — the ability to audit their data for bias
before it causes harm. No data science degree required.

## How It Works
1. Upload your CSV dataset
2. Select your target column and sensitive attribute
3. Optionally select a second sensitive attribute for intersectional analysis
4. EquiLens analyzes bias using SPD, DI, and EOD metrics
5. SHAP explains which features are causing the bias
6. Gemini translates everything into plain language
7. Intersectionality page reveals compounded disadvantage across identity combinations
8. What-if simulator lets you drop features and see the bias impact in real time
9. Download a full PDF audit report

## Real User Story
Priya runs an NGO in Pune distributing scholarships.
She uploads her dataset, selects gender as the sensitive
attribute and caste as the intersect attribute. She
discovers that lower-caste girls are approved at 8% —
far below the 34% rate for upper-caste boys. A Disparate
Impact of 0.24, well below the legal threshold of 0.8.
Gemini explains this in plain language and suggests fixes.
Priya downloads the audit report and shares it with her
board. All in under 5 minutes.

## Tech Stack
- **Backend:** FastAPI (Python)
- **Bias Metrics:** SPD, Disparate Impact, Equalized Odds
- **Explainability:** SHAP (TreeExplainer)
- **AI Layer:** Gemini 2.5 Flash (explanation + audience toggle + what-if analysis)
- **Frontend:** Single-page HTML/CSS/JS with Chart.js
- **PDF Export:** ReportLab
- **Deployment:** Render

## Fairness Metrics
| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Disparate Impact | < 0.8 = biased | Legal standard (EEOC 4/5ths rule) |
| Statistical Parity Difference | > 0.1 = biased | Outcome gap between groups |
| Equalized Odds | > 0.1 = biased | Error rate gap between groups |

## What Makes EquiLens Different
Every existing tool — IBM AI Fairness 360, Fairlearn,
Aequitas — outputs p-values and confusion matrices that
only data scientists can interpret. EquiLens translates
those results into plain language tuned to who's reading:
- **Student** → learning-oriented explanation
- **NGO worker** → policy implication
- **Policy maker** → legal risk framing

## SDG Alignment
EquiLens directly addresses UN Sustainable Development
Goal 10 — Reduced Inequalities. By making AI fairness
auditing accessible to organizations without data science
teams, we prevent algorithmic discrimination before it
reaches production.

---

## ✅ What's Done

### Backend
- [x] FastAPI server with CORS middleware
- [x] CSV upload and parsing via `/analyze` endpoint
- [x] Bias metrics computed locally: SPD, Disparate Impact, Equalized Odds (real TPR/FPR per group)
- [x] SHAP feature importance via `explainer.py`
- [x] Model training and evaluation via `trainer.py` (RandomForest, ROC + calibration curves)
- [x] Gemini 2.5 Flash integration for plain-language explanations (`gemini_client.py`)
- [x] Audience toggle: NGO worker / Student / Policy maker
- [x] Graceful fallback explanation when Gemini quota is exhausted or API key is invalid
- [x] Fallback explanation is dynamic — based on actual CSV data, not hardcoded
- [x] Fallback responses marked with `*` so developers know Gemini is not responding
- [x] Smart label decoding: encoded columns (0/1/2...) mapped to human-readable names
- [x] String target column support (e.g. "yes"/"no", "hired"/"rejected", ">50K"/"<=50K")
- [x] Frontend served via FastAPI static files mount (no CORS issues)
- [x] Real intersectionality computation via `compute_intersectionality()` in `analyzer.py`
- [x] Cross-group approval rates computed for every (col1 × col2) subgroup combination
- [x] Cells with fewer than 10 samples excluded and marked null to avoid misleading statistics
- [x] Correct integer→label decoding for multi-value encoded columns (race: 0–4)
- [x] Optional `sensitive_col_2` parameter on `/analyze` endpoint
- [x] Real Equalized Odds via `compute_eod()` — true TPR/FPR difference across groups
- [x] `/whatif` endpoint — retrain model on reduced feature set and measure bias delta
- [x] `/whatif/features` endpoint — returns available features from cached session
- [x] In-memory session cache — dataset cached after `/analyze` so `/whatif` doesn't require re-upload
- [x] What-if delta computed correctly: SPD/DI from model predictions, EOD from true labels
- [x] Gemini explains what-if results in plain language (`explain_whatif` in `gemini_client.py`)
- [x] PDF audit report generation via ReportLab (`pdf_exporter.py`)
- [x] PDF includes: verdict banner, metric scorecards, group approval table, SHAP bars, Gemini explanation, regulation compliance table

### Frontend
- [x] Single-page app with sidebar navigation
- [x] Overview page: score cards (DI, SPD, severity), approval rate chart, group comparison table
- [x] Fairness metrics page: metric bars, calibration curve, ROC by group
- [x] Calibration and ROC curves labeled with real group names (Female/Male, not 0/1)
- [x] Explainability page: SHAP feature importance bars with proxy variable detection
- [x] Intersectionality page: real sex × race heatmap from backend data
- [x] Intersectionality subgroup table: all cross-group combinations ranked worst → best
- [x] Intersectionality insight: AI text identifying most/least disadvantaged subgroup with count
- [x] Graceful fallback to single-attribute view when second column not selected
- [x] "Intersect with" dropdown auto-populated from CSV headers
- [x] Auto-detects race/ethnicity/caste as second sensitive attribute
- [x] Remediation page: before/after radar charts, recommended steps from backend
- [x] Audit report page: structured report with key findings and copy-to-clipboard
- [x] PDF download from audit report page
- [x] Domain switcher: Hiring / Credit / Healthcare demo presets
- [x] Auto-detect target and sensitive columns from uploaded CSV headers
- [x] Drag-and-drop CSV upload with column auto-population in dropdowns
- [x] Gemini audience toggle (NGO / Student / Policy maker)
- [x] Regulation compliance pills (EEOC, EU AI Act, GDPR)
- [x] What-if simulator: feature checkboxes, before/after radar charts, delta cards, Gemini explanation

---

## 🚧 What's Pending

### Frontend
- [ ] Audience toggle re-fetches explanation from backend without re-uploading CSV
- [ ] Intersectionality: sample size tooltip on null/sparse cells
- [ ] Mobile responsive layout
- [ ] Loading skeletons instead of spinner
- [ ] Inline error messages instead of `alert()` popups

### Deployment
- [ ] Environment variable management for production (`.env` → Cloud Secrets)

---

## Setup
```bash
git clone https://github.com/CheerathAniketh/Solution-Challenge
cd Solution-Challenge/backend
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` folder:
```
GEMINI_API_KEY=your_api_key_here
```

Run the server:
```bash
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

## License
MIT
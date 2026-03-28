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
- Backend: FastAPI (Python)
- Bias Metrics: SPD, Disparate Impact, Equalized Odds
- Explainability: SHAP
- AI Layer: Gemini API (explanation + audience toggle)
- Frontend: Single-page HTML/CSS/JS with Chart.js
- Deployment: Google Cloud Run

## Fairness Metrics
| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Disparate Impact | < 0.8 = biased | Legal standard (EEOC) |
| Statistical Parity Difference | > 0.1 = biased | Outcome gap between groups |
| Equalized Odds | > 0.1 = biased | Error rate gap between groups |

## What Makes EquiLens Different
Every existing tool — IBM AI Fairness 360, Fairlearn,
Aequitas — outputs p-values and confusion matrices that
only data scientists can interpret. EquiLens translates
those results into plain language tuned to who's reading:
- Student → learning-oriented explanation
- NGO worker → policy implication
- Policy maker → legal risk framing

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
- [x] Bias metrics computed locally: SPD, Disparate Impact, Equalized Odds (estimated)
- [x] SHAP feature importance via `explainer.py`
- [x] Model training and evaluation via `trainer.py`
- [x] Gemini API integration for plain-language explanations (`gemini_client.py`)
- [x] Audience toggle: NGO worker / Student / Policy maker
- [x] Graceful fallback explanation when Gemini quota is exhausted or API key is invalid
- [x] Fallback explanation is dynamic — based on actual CSV data, not hardcoded
- [x] Fallback responses marked with `*` so developers know Gemini is not responding
- [x] Smart label decoding: encoded columns (0/1/2...) mapped to human-readable names
- [x] String target column support (e.g. "yes"/"no", "hired"/"rejected")
- [x] Frontend served via FastAPI static files mount (no CORS issues)
- [x] Real intersectionality computation via `compute_intersectionality()` in `analyzer.py`
- [x] Cross-group approval rates computed for every (col1 × col2) subgroup combination
- [x] Cells with fewer than 10 samples excluded and marked null to avoid misleading statistics
- [x] Correct integer→label decoding for multi-value encoded columns (race: 0–4)
- [x] Optional `sensitive_col_2` parameter on `/analyze` endpoint

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
- [x] Domain switcher: Hiring / Credit / Healthcare demo presets
- [x] Auto-detect target and sensitive columns from uploaded CSV headers
- [x] Drag-and-drop CSV upload with column auto-population in dropdowns
- [x] Gemini audience toggle (NGO / Student / Policy maker)
- [x] Regulation compliance pills (EEOC, EU AI Act, GDPR)

---

## 🚧 What's Pending

### Backend
- [ ] `/whatif` endpoint — simulate bias impact of dropping specific features
- [ ] Equalized Odds computed properly (currently estimated from SPD)
- [ ] PDF export of audit report
- [ ] Authentication / API key management for multi-user deployments
- [ ] Rate limiting and input validation on file uploads

### Frontend
- [ ] What-if simulator page (UI exists, backend endpoint pending)
- [ ] Audience toggle re-fetches explanation from backend (currently only works for uploaded CSVs)
- [ ] Intersectionality: show sample size tooltip on null/sparse cells
- [ ] Mobile responsive layout (sidebar hidden on small screens but content not fully optimized)
- [ ] Loading skeletons instead of spinner during analysis
- [ ] Error messages shown inline instead of `alert()` popups

### Deployment
- [ ] Docker image finalized and tested
- [ ] Google Cloud Run deployment
- [ ] Environment variable management for production (`.env` → Cloud Secrets)
- [ ] Live demo URL

---

## Setup
```bash
git clone https://github.com/CheerathAniketh/EquiLens-AI
cd EquiLens-AI/backend
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
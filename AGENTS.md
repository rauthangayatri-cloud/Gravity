# Smart Campus Student Success Platform
Hackathon project (KPMG Smart Campus Analytics). Solo build.
Pipeline: 7 raw tables -> clean + unified student view -> Success Score ->
academic and placement risk models -> SHAP explanations -> priority
worklist -> playbook actions -> simulated outcome tracking -> React dashboard.

## Rules
- Do only what the current prompt asks. No extra features, models or pages.
- Python 3.11+, pandas, scikit-learn, XGBoost, SHAP. API: FastAPI.
  Frontend: React + Vite + Recharts.
- Fixed random seed (42) everywhere. Everything must be reproducible.
- Nothing trains or does heavy computation at runtime. Precompute to
  data/outputs/.
- Data is SYNTHETIC. Never claim real-world accuracy, fairness or causation
  in code comments, UI text or docs.
- Windows 11, PowerShell, no GPU, 16 GB RAM. Keep it lightweight.
- Every module gets short docstrings and basic tests. All tests must pass
  before finishing.
- Never leak outcome columns into model features.

## Three separate concepts (never mix them)
1. Success Score (0-100): current state, transparent weighted formula.
2. Risk probabilities: academic risk and placement risk, from ML.
3. Priority score: risk x severity x trend multiplier, for the worklist.

## Structure
data/raw, data/clean, data/outputs
src/ generate_data.py clean.py features.py score.py train.py segment.py
     priority.py build_outputs.py
api/main.py, web/, docs/, tests/, models/

## Success Score weights
Academic 35, Attendance 20, LMS 15, Placement readiness 15,
Engagement+skills 10, Feedback 5. Missing data: reweight over available
components; confidence = share of weight available; below 50% show
"Insufficient data".

## Run order
python -m src.generate_data
python -m src.clean
python -m src.build_outputs --unified data/clean/unified.csv --outcomes data/clean/outcomes.csv
uvicorn api.main:app --reload
cd web; npm install; npm run dev

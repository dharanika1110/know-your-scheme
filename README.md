# YojanaSetu — Government Scheme Eligibility + Coverage-Gap Intelligence

Two linked modules, both on real data (no synthetic data anywhere):

1. **Check My Eligibility** — rule-based matcher against 30 major Central
   Government schemes, using their real, publicly documented eligibility
   criteria (age, income, category, occupation, BPL/disability status).

2. **District Policy Insights** — real ML analytics (K-Means clustering +
   Random Forest regression, with a proper train/test split and reported
   R²/MAE) on India's NFHS-5 government health survey: 341 real districts
   x 11 real socioeconomic/scheme-coverage indicators. Flags districts
   where actual scheme coverage (health insurance, clean cooking fuel) is
   below what the district's socioeconomic profile predicts — a proxy for
   low awareness / under-enrollment, directly tied to the original problem
   statement.

Data sources:
- Schemes: manually compiled from official scheme criteria (PM-KISAN, PMJAY,
  PMAY, Ujjwala Yojana, etc.) — simplified for demonstration, verify exact
  current terms on each scheme's official page before relying on this tool.
- NFHS-5 district data: Ministry of Health & Family Welfare, Government of
  India, via https://github.com/pratapvardhan/NFHS-5 (CC-BY-4.0)

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy for a public link
1. Push this whole folder (including `data/`) to a new GitHub repo.
2. Go to https://share.streamlit.io -> "New app".
3. Point it at your repo, branch, and `app.py`.
4. Deploy -> you get a URL like https://your-app-name.streamlit.app

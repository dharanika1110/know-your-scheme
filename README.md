# Know Your Scheme

Helps people discover which Indian government schemes (Central + State) they're eligible for,
based on a short multi-step questionnaire.

## Files
- `app.py` — the Streamlit app
- `states_uts.py` — master list of India's states/UTs used for filtering
- `process_data.py` — one-time script that turns raw scheme text into structured,
  filterable fields (state, age range, income ceiling, gender, category, occupation).
  Re-run this if you get a new/updated raw scheme CSV.
- `schemes_processed.csv` — the structured output used by the app (already generated)
- `requirements.txt` — Python dependencies

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (include `schemes_processed.csv`).
2. Go to share.streamlit.io, connect the repo, set main file to `app.py`.
3. Deploy.

## Known limitations (be upfront with users about these)
- Eligibility fields (age, income, gender, caste category, occupation) were extracted
  from free-text using rule-based pattern matching, not manually verified per scheme.
  Roughly 250/3397 schemes have a detected age range, ~350 an income figure — the rest
  fall back to "no restriction detected," which is treated as a neutral/partial match,
  not a guarantee.
- ~50 schemes couldn't be matched to a specific state and are excluded from
  state-specific filtering (tagged "Unknown").
- The match % is a heuristic score to help people discover relevant schemes quickly —
  not a legal determination of eligibility. Always tell users to verify on the
  official portal before applying.

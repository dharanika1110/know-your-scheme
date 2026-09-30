# Know Your Scheme
Find every government scheme you are eligible for. Streamlit web app + ML evaluation (3,397 schemes).

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app (Find Schemes, EDA, Model Evaluation tabs) |
| `matcher.py` | Eligibility filtering and match-score engine |
| `states_uts.py` | List of all states and union territories |
| `train_model.py` | 80/20 train-test split, evaluation, writes `metrics.json` |
| `metrics.json` | Saved evaluation results shown in the app |
| `schemes_processed.csv` | Scheme dataset |
| `requirements.txt` | Dependencies for Streamlit Cloud |

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Re-run evaluation
    python train_model.py

## Method
- **Train/test split:** 2,717 training schemes (80%) and 680 unseen test schemes (20%), stratified by category.
- **Model:** TF-IDF + Logistic Regression predicts a scheme's category from its text.
- **Match score:** eligibility criteria verified (age, income, gender, category, occupation), location, interests and text relevance.

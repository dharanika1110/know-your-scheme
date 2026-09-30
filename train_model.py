"""Train / test evaluation for Know Your Scheme.
import warnings; warnings.filterwarnings("ignore")
Run:  python train_model.py     ->  writes metrics.json (shown in the app's 'Model Evaluation' tab)
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from matcher import load_data, match, OCCUPATIONS

df = load_data()
X, y = df["text"], df["primary_category"]

# 1) 80% training data / 20% unseen test data (stratified so every category is in both)
tr_idx, te_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
model = make_pipeline(TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2), sublinear_tf=True),
                      LogisticRegression(max_iter=2000, C=10))
model.fit(X[tr_idx], y[tr_idx])
pred_tr, pred_te = model.predict(X[tr_idx]), model.predict(X[te_idx])
cv = cross_val_score(model, X[tr_idx], y[tr_idx], cv=5)
rep = classification_report(y[te_idx], pred_te, output_dict=True, zero_division=0)
baseline = float((y[te_idx] == y[tr_idx].mode()[0]).mean())

# 2) Recommender check on the unseen test schemes:
#    build a profile that is eligible for each test scheme and see whether the scheme is ranked in the top-K
rng = np.random.default_rng(42)
states = sorted(s for s in df["state"].unique() if s not in ("All India", "Unknown"))
ranks = []
for i in te_idx:
    r = df.loc[i]
    prof = {"state": r.state if r.level == "State" else rng.choice(states),
            "age": int((r.min_age if r.min_age == r.min_age else 25) + (r.max_age if r.max_age == r.max_age else 35)) // 2
                   if (r.min_age == r.min_age or r.max_age == r.max_age) else 30,
            "income": int(r.max_income * 0.5) if r.max_income == r.max_income else 0,
            "gender": "Female" if r.gender == "Female" else rng.choice(["Female", "Male"]),
            "caste": next((c for c in r.caste_set if c not in ("General/Any", "Minority")), "General"),
            "minority": "Minority" in r.caste_set,
            "occupations": sorted(r.occ_set)[:1] if r.occ_set else [],
            "interests": [r.primary_category]}
    res = match(df, prof)
    pos = np.where(res.index == i)[0]
    ranks.append(int(pos[0]) + 1 if len(pos) else 10**6)
ranks = np.array(ranks)
hit = lambda k: round(float((ranks <= k).mean()), 4)

metrics = {
    "n_total": len(df), "n_train": len(tr_idx), "n_test": len(te_idx), "n_classes": int(y.nunique()),
    "train_accuracy": round(accuracy_score(y[tr_idx], pred_tr), 4),
    "test_accuracy": round(accuracy_score(y[te_idx], pred_te), 4),
    "cv_accuracy_mean": round(float(cv.mean()), 4), "cv_accuracy_std": round(float(cv.std()), 4),
    "test_macro_f1": round(f1_score(y[te_idx], pred_te, average="macro"), 4),
    "test_weighted_f1": round(f1_score(y[te_idx], pred_te, average="weighted"), 4),
    "majority_baseline_accuracy": round(baseline, 4),
    "per_class": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score")} | {"support": int(v["support"])}
                  for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")},
    "recommender": {"hit_at_10": hit(10), "hit_at_50": hit(50), "hit_at_100": hit(100),
                    "median_rank": int(np.median(ranks[ranks < 10**6])) if (ranks < 10**6).any() else None,
                    "n_profiles": int(len(ranks))},
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
print(classification_report(y[te_idx], pred_te, zero_division=0))

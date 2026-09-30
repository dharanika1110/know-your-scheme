"""Eligibility matching + match-score engine (shared by app.py and train_model.py)."""
import numpy as np
import pandas as pd

OCCUPATIONS = ["Student", "Laborer/Worker", "Entrepreneur/MSME", "Farmer",
               "Person with Disability", "Widow", "Senior Citizen",
               "Artisan/Craftsman", "Unemployed", "Fisherman"]
CASTES = ["General", "SC", "ST", "OBC", "EWS", "Minority"]
GENDERS = ["Female", "Male", "Other"]


def load_data(path="schemes_processed.csv"):
    df = pd.read_csv(path)
    for c in ["details", "benefits", "eligibility", "application", "documents", "tags", "occupation_tags"]:
        df[c] = df[c].fillna("")
    df["occ_set"] = df["occupation_tags"].apply(lambda s: {t.strip() for t in s.split(",") if t.strip()})
    df["caste_set"] = df["caste_category"].fillna("General/Any").apply(lambda s: {t.strip() for t in s.split(",")})
    # text used by the ML model (category name is NOT included, to avoid label leakage)
    df["text"] = df["scheme_name"] + " " + df["tags"] + " " + df["details"].str[:1500]
    return df


def match(df, prof, sim=None, pred_cat=None):
    """Return eligible schemes with a 0-100 match score, best first.
    prof keys: state, age, income (0 = not given), gender, caste, occupations, interests."""
    n = len(df)
    ok = np.ones(n, bool)
    crit = np.zeros(n)      # criteria the scheme specifies
    hit = np.zeros(n)       # criteria verified against the user's profile

    central = (df["level"] == "Central").values
    same_state = (df["state"] == prof["state"]).values
    ok &= central | same_state

    age = prof.get("age")
    if age:
        mn, mx = df["min_age"], df["max_age"]
        bad = ((mn.notna()) & (age < mn)) | ((mx.notna()) & (age > mx))
        ok &= ~bad.values
        spec = (mn.notna() | mx.notna()).values
        crit += spec; hit += spec
    else:
        crit += (df["min_age"].notna() | df["max_age"].notna()).values

    inc = prof.get("income")
    has_inc = df["max_income"].notna().values
    if inc:
        ok &= ~(has_inc & (inc > df["max_income"].fillna(0).values))
        crit += has_inc; hit += has_inc
    else:
        crit += has_inc

    g = df["gender"].values
    gender_specific = g != "Any"
    ok &= ~(gender_specific & (g != prof.get("gender")))
    crit += gender_specific; hit += gender_specific

    caste = prof.get("caste", "General")
    cs = df["caste_set"]
    caste_specific = cs.apply(lambda s: "General/Any" not in s).values
    caste_ok = cs.apply(lambda s: caste in s or ("Minority" in s and prof.get("minority", False))).values
    ok &= ~(caste_specific & ~caste_ok)
    crit += caste_specific; hit += caste_specific

    occ = set(prof.get("occupations", []))
    occ_specific = df["occ_set"].apply(bool).values
    occ_hit = df["occ_set"].apply(lambda s: bool(s & occ)).values
    crit += occ_specific; hit += occ_hit & occ_specific

    elig = np.where(crit > 0, hit / np.maximum(crit, 1), 0.5) * 50
    loc = np.where(same_state, 15, 10)
    interests = set(prof.get("interests", []))
    if pred_cat:
        interests.add(pred_cat)
    cat = df["primary_category"].isin(interests).values * 20 if interests else np.zeros(n)
    txt = np.zeros(n)
    if sim is not None and sim.max() > 0:
        txt = 15 * sim / sim.max()
    max_pts = 65 + (20 if interests else 0) + (15 if txt.any() else 0)
    out = df.assign(match=np.round((elig + loc + cat + txt) / max_pts * 100).astype(int))
    return out[ok].sort_values("match", ascending=False)

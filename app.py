"""
YojanaSetu — Bridging Citizens to Government Schemes
A Data-Driven System for Scheme Discovery and Coverage-Gap Analysis

Two linked modules:
1. CITIZEN CHECKER   — rule-based matching against real, publicly documented
   eligibility criteria for 30 major Central Government schemes.
2. POLICY INSIGHTS   — real ML analytics (K-Means clustering + Random Forest
   regression) on India's National Family Health Survey (NFHS-5) district
   data: 341 real districts x 11 real socioeconomic/scheme-coverage
   indicators = 3,700+ real data points, sourced from the Ministry of
   Health & Family Welfare, Government of India.
   Source: https://github.com/pratapvardhan/NFHS-5 (CC-BY-4.0)

Data Science Lifecycle:
Business Problem -> Data Collection -> Data Cleaning -> EDA ->
Feature Engineering -> Modeling (Clustering + Regression) -> Evaluation ->
Explainability -> Citizen Recommendation + Policy Insight
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="YojanaSetu", page_icon="🏛️", layout="wide")

DATA_DIR = "."

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir",
    "Ladakh", "Puducherry", "Chandigarh",
]

# ----------------------------------------------------------------------
# STEP 1 + 2: DATA COLLECTION & CLEANING
# ----------------------------------------------------------------------


@st.cache_data
def load_schemes():
    df = pd.read_csv(f"{DATA_DIR}/schemes.csv")
    df["min_age"] = pd.to_numeric(df["min_age"], errors="coerce").fillna(0)
    df["max_age"] = pd.to_numeric(df["max_age"], errors="coerce").fillna(200)
    df["max_annual_income"] = pd.to_numeric(df["max_annual_income"], errors="coerce")
    return df


@st.cache_data
def load_nfhs():
    df = pd.read_csv(f"{DATA_DIR}/nfhs_district_wide.csv")
    df = df.dropna()
    rename = {
        "Female population age 6 years and above who ever attended school (%)": "school_attendance",
        "Households using clean fuel for cooking3 (%)": "clean_fuel",
        "Households with any usual member covered under a health insurance/financing scheme (%)": "health_insurance",
        "Institutional births (%)": "institutional_births",
        "Institutional births in public facility (%)": "institutional_births_public",
        "Population below age 15 years (%)": "pop_under_15",
        "Population living in households that use an improved sanitation facility2 (%)": "sanitation",
        "Population living in households with an improved drinking-water source1 (%)": "drinking_water",
        "Population living in households with electricity (%)": "electricity",
        "Women who are literate4 (%)": "women_literacy",
        "Women with 10 or more years of schooling (%)": "women_schooling",
    }
    df = df.rename(columns=rename)
    return df.reset_index(drop=True)


schemes = load_schemes()
nfhs = load_nfhs()

FEATURES = [
    "school_attendance", "pop_under_15", "sanitation", "drinking_water",
    "electricity", "women_literacy", "women_schooling",
    "institutional_births", "institutional_births_public",
]
TARGETS = {"health_insurance": "Health Insurance Coverage (PMJAY-relevant)",
           "clean_fuel": "Clean Cooking Fuel Coverage (Ujjwala-relevant)"}

# ----------------------------------------------------------------------
# STEP 3 + 4: FEATURE ENGINEERING & MODELS
# ----------------------------------------------------------------------


@st.cache_resource
def train_cluster_model(_df):
    X = _df[FEATURES]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    return km, scaler, labels


@st.cache_resource
def train_gap_models(_df):
    models = {}
    for target in TARGETS:
        X = _df[FEATURES]
        y = _df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)
        r2 = r2_score(y_test, preds_test)
        mae = mean_absolute_error(y_test, preds_test)
        preds_full = model.predict(X)
        models[target] = {
            "model": model, "r2": r2, "mae": mae,
            "predicted_full": preds_full,
            "importances": dict(zip(FEATURES, model.feature_importances_)),
        }
    return models


km_model, scaler, cluster_labels = train_cluster_model(nfhs)
nfhs["cluster"] = cluster_labels
gap_models = train_gap_models(nfhs)
for target, m in gap_models.items():
    nfhs[f"{target}_predicted"] = m["predicted_full"]
    nfhs[f"{target}_gap"] = nfhs[target] - nfhs[f"{target}_predicted"]

# Label clusters by their average profile (low -> high overall development)
cluster_avg_dev = nfhs.groupby("cluster")[FEATURES].mean().mean(axis=1).sort_values()
cluster_rank = {c: i for i, c in enumerate(cluster_avg_dev.index)}
CLUSTER_NAMES = {
    0: "🔴 High-need — low infrastructure & coverage",
    1: "🟠 Developing — moderate infrastructure",
    2: "🟡 Improving — above-average coverage",
    3: "🟢 Well-served — strong infrastructure & coverage",
}
nfhs["cluster_rank"] = nfhs["cluster"].map(cluster_rank)
nfhs["cluster_label"] = nfhs["cluster_rank"].map(CLUSTER_NAMES)

# ----------------------------------------------------------------------
# MATCHING LOGIC (citizen checker)
# ----------------------------------------------------------------------


def check_eligibility(age, gender, income, category, occupation, bpl, disability, state):
    matches, close_misses = [], []
    for _, s in schemes.iterrows():
        reasons_failed = []

        if not (s["min_age"] <= age <= s["max_age"]):
            reasons_failed.append(f"age must be {int(s['min_age'])}-{int(s['max_age']) if s['max_age']<200 else '∞'}")

        if pd.notna(s["max_annual_income"]) and income > s["max_annual_income"]:
            reasons_failed.append(f"income must be ≤ ₹{s['max_annual_income']:,.0f}")

        cats = [c.strip() for c in str(s["eligible_category"]).split("/")]
        cat_ok = "All" in cats or category in cats or f"{category}-BPL" in cats or any(
            "BPL" in c and bpl == "Yes" for c in cats
        )
        if not cat_ok:
            reasons_failed.append(f"category must be one of {s['eligible_category']}")

        if s["gender"] != "All" and s["gender"] != gender:
            reasons_failed.append(f"gender must be {s['gender']}")

        occ_req = str(s["occupation"])
        if occ_req not in ("Any",) and occ_req.lower() not in occupation.lower() and occupation.lower() not in occ_req.lower():
            reasons_failed.append(f"occupation should be '{occ_req}'")

        if s["bpl_required"] == "Yes" and bpl != "Yes":
            reasons_failed.append("must hold a BPL card")

        if s["disability_required"] == "Yes" and disability != "Yes":
            reasons_failed.append("must have a recognised disability")

        record = s.to_dict()
        if not reasons_failed:
            matches.append(record)
        elif len(reasons_failed) <= 1:
            record["missed_on"] = reasons_failed[0]
            close_misses.append(record)

    return matches, close_misses


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("🏛️ YojanaSetu")
st.caption(
    "Bridging citizens to the government schemes they qualify for — and helping "
    "policymakers see where scheme awareness/coverage is falling short."
)

tab1, tab2 = st.tabs(["🧑 Check My Eligibility", "📊 District Policy Insights"])

# ============================== TAB 1 ==============================
with tab1:
    with st.expander("📋 About this checker"):
        st.markdown(
            f"""
Matches you against **{len(schemes)} major Central Government schemes**, using
their real, publicly documented eligibility criteria (age, income, category,
occupation, BPL/disability status). Criteria are simplified for demonstration —
always verify exact current terms on the scheme's official page before applying.
            """
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=0, max_value=100, value=28)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with c2:
        income = st.number_input("Annual Household Income (₹)", min_value=0, value=150000, step=10000)
        category = st.selectbox("Social Category", ["General", "OBC", "SC", "ST"])
    with c3:
        state = st.selectbox("State / UT", INDIAN_STATES)
        occupation = st.selectbox(
            "Occupation",
            ["Any", "Farmer", "Student", "Unemployed", "Small/micro entrepreneur",
             "Rural unskilled worker", "Senior citizen", "Homeowner", "Artisan",
             "Rural woman", "Salaried"],
        )

    c4, c5 = st.columns(2)
    with c4:
        bpl = st.radio("Do you hold a BPL card?", ["No", "Yes"], horizontal=True)
    with c5:
        disability = st.radio("Do you have a recognised disability?", ["No", "Yes"], horizontal=True)

    if st.button("🔍 FIND MY SCHEMES", type="primary", width='stretch'):
        matches, close_misses = check_eligibility(
            age, gender, income, category, occupation, bpl, disability, state
        )

        st.divider()
        st.subheader(f"✅ You appear eligible for {len(matches)} scheme(s)")
        if matches:
            for m in matches:
                with st.container(border=True):
                    st.markdown(f"**{m['scheme_name']}** — *{m['ministry']}*")
                    st.write(m["benefit_summary"])
                    st.caption(f"Category: {m['category_tag']} · [Official page]({m['official_link']})")
        else:
            st.info("No exact matches found among the 30 schemes in this demo. See close matches below.")

        if close_misses:
            st.subheader(f"🟡 {len(close_misses)} scheme(s) you're close to qualifying for")
            for m in close_misses:
                with st.container(border=True):
                    st.markdown(f"**{m['scheme_name']}** — *{m['ministry']}*")
                    st.write(f"Missed on: {m['missed_on']}")
                    st.caption(f"[Official page]({m['official_link']})")

# ============================== TAB 2 ==============================
with tab2:
    st.markdown(
        f"Real ML analysis on **{len(nfhs):,} real districts × {len(FEATURES)+len(TARGETS)} real indicators** "
        f"from India's NFHS-5 government health survey (Govt of India, Ministry of Health & Family Welfare)."
    )

    with st.expander("📈 Model evaluation (train/test, held-out districts)"):
        for target, label in TARGETS.items():
            m = gap_models[target]
            mc1, mc2 = st.columns(2)
            mc1.metric(f"{label} — R²", f"{m['r2']:.2f}")
            mc2.metric(f"{label} — MAE", f"{m['mae']:.1f} pts")
        st.caption(
            "R² and MAE computed on a held-out 20% test split of real districts — "
            "not seen during training."
        )

    state_pick = st.selectbox("Select a State", sorted(nfhs["State"].unique()))
    districts_in_state = nfhs[nfhs["State"] == state_pick]
    district_pick = st.selectbox("Select a District", sorted(districts_in_state["District"].unique()))

    row = nfhs[(nfhs["State"] == state_pick) & (nfhs["District"] == district_pick)].iloc[0]

    st.divider()
    st.subheader(f"{district_pick}, {state_pick}")
    st.markdown(f"**Development profile:** {row['cluster_label']}")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Electricity access", f"{row['electricity']:.1f}%")
    mc2.metric("Women's literacy", f"{row['women_literacy']:.1f}%")
    mc3.metric("Sanitation access", f"{row['sanitation']:.1f}%")

    st.markdown("#### Scheme coverage: actual vs. model-expected")
    for target, label in TARGETS.items():
        actual = row[target]
        predicted = row[f"{target}_predicted"]
        gap = row[f"{target}_gap"]
        cols = st.columns([2, 1, 1, 1])
        cols[0].write(f"**{label}**")
        cols[1].metric("Actual", f"{actual:.1f}%")
        cols[2].metric("Model-expected", f"{predicted:.1f}%")
        if gap < -8:
            cols[3].error(f"{gap:+.1f} gap")
        elif gap < -3:
            cols[3].warning(f"{gap:+.1f} gap")
        else:
            cols[3].success(f"{gap:+.1f} gap")

    worst_gap_target = min(TARGETS, key=lambda t: row[f"{t}_gap"])
    worst_gap = row[f"{worst_gap_target}_gap"]
    if worst_gap < -8:
        st.error(
            f"🚩 **{district_pick}** shows a significant coverage gap in "
            f"**{TARGETS[worst_gap_target]}** — the district's socioeconomic profile "
            f"predicts higher coverage than what's actually recorded. This pattern "
            f"often indicates low scheme awareness/enrollment rather than ineligibility, "
            f"and is a strong candidate for targeted outreach."
        )
    elif worst_gap < -3:
        st.warning(
            f"⚠️ Moderate coverage gap detected in **{TARGETS[worst_gap_target]}** "
            f"relative to what the district's profile predicts."
        )
    else:
        st.success("✅ Coverage roughly matches or exceeds what this district's profile predicts.")

    st.markdown("#### What drives coverage? (Model feature importance)")
    imp_target = st.selectbox("Show importances for:", list(TARGETS.keys()), format_func=lambda t: TARGETS[t])
    imp = gap_models[imp_target]["importances"]
    imp_df = pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())}).sort_values(
        "importance", ascending=False
    )
    st.bar_chart(imp_df.set_index("feature"))

    st.markdown("#### All districts in this state — coverage gap overview")
    state_df = nfhs[nfhs["State"] == state_pick][
        ["District", "cluster_label"] + list(TARGETS.keys()) + [f"{t}_gap" for t in TARGETS]
    ].sort_values(f"{list(TARGETS.keys())[0]}_gap")
    st.dataframe(state_df, width='stretch', hide_index=True)

st.divider()
with st.expander("ℹ️ About the data science pipeline"):
    st.markdown(
        f"""
**Citizen Checker data:** {len(schemes)} major Central Government schemes, real
publicly documented eligibility criteria (manually compiled and simplified for
demonstration — verify exact terms on official scheme pages).

**Policy Insights data:** {len(nfhs):,} real districts × {len(FEATURES)+len(TARGETS)}
real indicators from NFHS-5 (2019-21), India's official national health survey,
Ministry of Health & Family Welfare, Government of India. Source (CC-BY-4.0):
github.com/pratapvardhan/NFHS-5

**Pipeline:** Business Problem → Data Collection → Data Cleaning → EDA →
Feature Engineering → K-Means Clustering (district profiling) + Random Forest
Regression (coverage-gap prediction, evaluated on a held-out test split) →
Explainability (feature importance) → Citizen Recommendation + Policy Insight.

**No synthetic data is used anywhere in this app.**
        """
    )

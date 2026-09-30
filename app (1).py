import html, json, os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from states_uts import ALL_STATES_UTS
from matcher import load_data, match, OCCUPATIONS, CASTES, GENDERS

st.set_page_config(page_title="Know Your Scheme", page_icon="🇮🇳", layout="wide")
st.markdown("""<style>
.title{text-align:center;font-size:2.4rem;font-weight:800;color:#1B4332;margin-bottom:0}
.sub{text-align:center;color:#5B6B63;margin-bottom:1.2rem}
.card{background:#fff;border:1px solid #e3e8e4;border-radius:12px;padding:14px 18px;margin-top:10px;
      display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.name{font-weight:700;font-size:1.05rem;color:#1F2933}
.chip{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600;margin:6px 6px 0 0}
.c1{background:#FDE8D4;color:#B45309}.c2{background:#E0E7FF;color:#4338CA}
.badge{background:#15803d;color:#fff;border-radius:16px;padding:4px 12px;font-weight:700;white-space:nowrap}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading schemes...")
def get_data():
    return load_data(os.path.join(os.path.dirname(__file__), "schemes_processed.csv"))


@st.cache_resource(show_spinner="Preparing the recommendation model...")
def get_models(_df):
    """TF-IDF text search + category classifier (trained on all schemes for the live app)."""
    tf = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    X = tf.fit_transform(_df["text"])
    clf = make_pipeline(TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2), sublinear_tf=True),
                        LogisticRegression(max_iter=2000, C=10)).fit(_df["text"], _df["primary_category"])
    return tf, X, clf


def clean(t):
    return str(t).replace("$", "\\$")


df = get_data()
tf, X, clf = get_models(df)
ss = st.session_state
ss.setdefault("profile", None); ss.setdefault("shown", 10)

st.markdown('<div class="title">🇮🇳 Know Your Scheme</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Find every government scheme you\'re eligible for — in under a minute.</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🔍 Find Schemes", "📊 Data Insights (EDA)", "🤖 Model Evaluation"])

# ---------------- Tab 1: find schemes ----------------
with tab1:
    if ss.profile is None:
        with st.form("profile_form"):
            c1, c2, c3 = st.columns(3)
            state = c1.selectbox("State / UT", ALL_STATES_UTS, index=ALL_STATES_UTS.index("Tamil Nadu"))
            age = c2.number_input("Age", 0, 100, 25)
            gender = c3.selectbox("Gender", GENDERS)
            c4, c5, c6 = st.columns(3)
            income = c4.number_input("Annual family income (₹) — 0 to skip", 0, 10_000_000, 0, step=10000)
            caste = c5.selectbox("Category", CASTES)
            minority = c6.checkbox("I belong to a minority community")
            occ = st.multiselect("Occupation / situation", OCCUPATIONS)
            interests = st.multiselect("Areas of interest (optional)", sorted(df["primary_category"].unique()))
            need = st.text_input("Describe what you need (optional)", placeholder="e.g. scholarship for my daughter's college education")
            go = st.form_submit_button("Find my schemes", type="primary", width="stretch")
        if go:
            ss.profile = dict(state=state, age=int(age), gender=gender, income=int(income), caste=caste,
                              minority=minority, occupations=occ, interests=interests, need=need.strip())
            ss.shown = 10
            st.rerun()
    else:
        p = ss.profile
        sim, pred = None, None
        if p["need"]:
            sim = (X @ tf.transform([p["need"]]).T).toarray().ravel()
            pred = clf.predict([p["need"]])[0]
        res = match(df, p, sim=sim, pred_cat=pred)
        st.success(f"Found **{len(res)}** schemes matching your profile (out of {len(df)} total schemes in our database)")
        if pred:
            st.caption(f"🤖 Model detected the category of your need as: **{pred}**")
        if st.button("🔄 Start Over"):
            ss.profile = None; st.rerun()
        with st.expander("🔧 Refine results"):
            f1, f2, f3 = st.columns(3)
            lv = f1.multiselect("Level", ["Central", "State"], default=["Central", "State"])
            cats = f2.multiselect("Category", sorted(res["primary_category"].unique()))
            minm = f3.slider("Minimum match %", 0, 100, 0)
        res = res[res["level"].isin(lv) & (res["match"] >= minm)]
        if cats:
            res = res[res["primary_category"].isin(cats)]
        res = res.head(100)
        for _, r in res.head(ss.shown).iterrows():
            lvl = "Central — All India" if r.level == "Central" else f"State — {r.state}"
            st.markdown(f'<div class="card"><div><div class="name">{html.escape(r.scheme_name)}</div>'
                        f'<span class="chip c1">{html.escape(lvl)}</span><span class="chip c2">{html.escape(r.primary_category)}</span></div>'
                        f'<div class="badge">{r.match}% match</div></div>', unsafe_allow_html=True)
            with st.expander("View details"):
                for label, col in [("Details", "details"), ("Benefits", "benefits"), ("Eligibility", "eligibility"),
                                   ("How to apply", "application"), ("Documents required", "documents")]:
                    if r[col]:
                        st.markdown(f"**{label}**"); st.markdown(clean(r[col][:3000]))
                st.markdown(f"[Open on myScheme](https://www.myscheme.gov.in/schemes/{r.slug})")
        if len(res) > ss.shown and st.button("Show more schemes"):
            ss.shown += 10; st.rerun()
        if res.empty:
            st.info("No schemes match these filters. Try relaxing the refine options.")

# ---------------- Tab 2: EDA ----------------
with tab2:
    a, b, c = st.columns(3)
    a.metric("Total schemes", f"{len(df):,}")
    b.metric("Central schemes", int((df.level == "Central").sum()))
    c.metric("State / UT schemes", int((df.level == "State").sum()))
    st.subheader("Schemes by category"); st.bar_chart(df["primary_category"].value_counts())
    st.subheader("Schemes by state / UT"); st.bar_chart(df[df.state != "Unknown"]["state"].value_counts())
    st.subheader("Data completeness")
    st.dataframe(pd.DataFrame({"Schemes with this criterion": {
        "Age limit": int((df.min_age.notna() | df.max_age.notna()).sum()), "Income limit": int(df.max_income.notna().sum()),
        "Gender-specific": int((df.gender != "Any").sum()), "Caste/community-specific": int((df.caste_category != "General/Any").sum()),
        "Occupation tagged": int((df.occupation_tags != "").sum())}}))

# ---------------- Tab 3: model evaluation ----------------
with tab3:
    try:
        m = json.load(open(os.path.join(os.path.dirname(__file__), "metrics.json")))
    except FileNotFoundError:
        m = None; st.info("Run `python train_model.py` to generate metrics.json.")
    if m:
        st.markdown(f"**Category classifier** — {m['n_train']:,} training schemes (80%) and **{m['n_test']} unseen test schemes (20%)**, {m['n_classes']} categories.")
        k = st.columns(4)
        k[0].metric("Train accuracy", f"{m['train_accuracy']:.1%}"); k[1].metric("Test accuracy (unseen)", f"{m['test_accuracy']:.1%}")
        k[2].metric("Test macro F1", f"{m['test_macro_f1']:.2f}"); k[3].metric("Majority baseline", f"{m['majority_baseline_accuracy']:.1%}")
        st.caption(f"5-fold cross-validation accuracy: {m['cv_accuracy_mean']:.1%} ± {m['cv_accuracy_std']:.1%}")
        st.dataframe(pd.DataFrame(m["per_class"]).T, width="stretch")
        r = m["recommender"]
        st.markdown(f"**Recommender check on the unseen test schemes** — for each of {r['n_profiles']} held-out schemes, an eligible profile was generated and we checked whether the scheme was recommended.")
        k = st.columns(4)
        k[0].metric("Hit @ top 10", f"{r['hit_at_10']:.1%}"); k[1].metric("Hit @ top 50", f"{r['hit_at_50']:.1%}")
        k[2].metric("Hit @ top 100", f"{r['hit_at_100']:.1%}"); k[3].metric("Median rank", r["median_rank"])

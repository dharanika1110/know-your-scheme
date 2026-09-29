import streamlit as st
import pandas as pd

from states_uts import ALL_STATES_UTS

st.set_page_config(
    page_title="Know Your Scheme",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .stApp {
        background: linear-gradient(180deg, #FFF7ED 0%, #FAFAFA 15%);
    }
    .kys-hero {
        text-align: center;
        padding: 1.6rem 1rem 1rem 1rem;
    }
    .kys-hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #FF9933 0%, #138808 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kys-hero p {
        color: #555;
        font-size: 1.05rem;
    }
    .scheme-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        border: 1px solid #eee;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .scheme-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.10);
        border-color: #FFA94D;
    }
    .scheme-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.3rem;
    }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-central { background: #FFE8CC; color: #C2410C; }
    .badge-state { background: #DCFCE7; color: #15803D; }
    .badge-cat { background: #E0E7FF; color: #3730A3; }
    .match-score {
        float: right;
        background: #138808;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
    div[data-testid="stForm"] {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("schemes_processed.csv")
    return df

df = load_data()

CATEGORIES = sorted(df["primary_category"].dropna().unique().tolist())
OCCUPATIONS = ["Student", "Farmer", "Fisherman", "Laborer/Worker", "Entrepreneur/MSME",
               "Senior Citizen", "Person with Disability", "Unemployed",
               "Artisan/Craftsman", "Widow", "Other / Not sure"]
CASTE_OPTIONS = ["General", "SC", "ST", "OBC", "EWS", "Minority", "Not sure / Prefer not to say"]

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "saved_schemes" not in st.session_state:
    st.session_state.saved_schemes = set()

def go_next():
    st.session_state.step += 1

def go_back():
    st.session_state.step -= 1

def restart():
    st.session_state.step = 0
    st.session_state.answers = {}

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="kys-hero">
    <h1>🇮🇳 Know Your Scheme</h1>
    <p>Find every government scheme you're eligible for — in under a minute.</p>
</div>
""", unsafe_allow_html=True)

TOTAL_STEPS = 5

# ---------------------------------------------------------------------------
# STEP FLOW (multi-step form)
# ---------------------------------------------------------------------------
if st.session_state.step < TOTAL_STEPS:
    st.progress((st.session_state.step) / TOTAL_STEPS)
    step = st.session_state.step

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            if step == 0:
                st.subheader("📍 Where do you live?")
                state = st.selectbox("Your State / UT", ["-- Select --"] + ALL_STATES_UTS,
                                      index=(["-- Select --"] + ALL_STATES_UTS).index(
                                          st.session_state.answers.get("state", "-- Select --")))
                if st.button("Next ➜", use_container_width=True, disabled=(state == "-- Select --")):
                    st.session_state.answers["state"] = state
                    go_next()
                    st.rerun()

            elif step == 1:
                st.subheader("🎂 A little about you")
                age = st.number_input("Your age", min_value=0, max_value=110,
                                       value=st.session_state.answers.get("age", 25))
                gender = st.radio("Gender", ["Male", "Female", "Other"], horizontal=True,
                                   index=["Male", "Female", "Other"].index(
                                       st.session_state.answers.get("gender", "Male")))
                c1, c2 = st.columns(2)
                if c1.button("⟵ Back", use_container_width=True):
                    go_back(); st.rerun()
                if c2.button("Next ➜", use_container_width=True):
                    st.session_state.answers["age"] = age
                    st.session_state.answers["gender"] = gender
                    go_next()
                    st.rerun()

            elif step == 2:
                st.subheader("💰 Household income")
                income = st.number_input("Approximate annual family income (₹)", min_value=0,
                                          value=st.session_state.answers.get("income", 200000), step=10000)
                category = st.selectbox("Social category", CASTE_OPTIONS,
                                         index=CASTE_OPTIONS.index(
                                             st.session_state.answers.get("category", "General")))
                c1, c2 = st.columns(2)
                if c1.button("⟵ Back", use_container_width=True):
                    go_back(); st.rerun()
                if c2.button("Next ➜", use_container_width=True):
                    st.session_state.answers["income"] = income
                    st.session_state.answers["category"] = category
                    go_next()
                    st.rerun()

            elif step == 3:
                st.subheader("💼 What best describes you?")
                occupation = st.multiselect("Select all that apply", OCCUPATIONS,
                                             default=st.session_state.answers.get("occupation", []))
                disability = st.checkbox("I am a person with disability",
                                          value=st.session_state.answers.get("disability", False))
                c1, c2 = st.columns(2)
                if c1.button("⟵ Back", use_container_width=True):
                    go_back(); st.rerun()
                if c2.button("Next ➜", use_container_width=True):
                    st.session_state.answers["occupation"] = occupation
                    st.session_state.answers["disability"] = disability
                    go_next()
                    st.rerun()

            elif step == 4:
                st.subheader("🎯 Interested in any specific area?")
                interests = st.multiselect("Optional — narrow down your results",
                                            CATEGORIES,
                                            default=st.session_state.answers.get("interests", []))
                c1, c2 = st.columns(2)
                if c1.button("⟵ Back", use_container_width=True):
                    go_back(); st.rerun()
                if c2.button("🔍 Find My Schemes", use_container_width=True, type="primary"):
                    st.session_state.answers["interests"] = interests
                    go_next()
                    st.rerun()

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
else:
    ans = st.session_state.answers

    def score_row(row):
        score = 0
        max_score = 0

        # State match (Central schemes always match)
        max_score += 3
        if row["level"] == "Central" or row["state"] == ans["state"] or row["state"] == "Unknown":
            score += 3

        # Age
        max_score += 2
        if pd.notna(row["min_age"]) and pd.notna(row["max_age"]):
            if row["min_age"] <= ans["age"] <= row["max_age"]:
                score += 2
        else:
            score += 1  # unknown -> neutral partial credit

        # Income
        max_score += 2
        if pd.notna(row["max_income"]):
            if ans["income"] <= row["max_income"]:
                score += 2
        else:
            score += 1

        # Gender
        max_score += 1
        if row["gender"] == "Any" or row["gender"] == ans["gender"]:
            score += 1

        # Caste/category
        max_score += 2
        cats = str(row["caste_category"]).split(",")
        if ans["category"] in ("Not sure / Prefer not to say",) or "General/Any" in cats or ans["category"] in cats \
                or (ans["category"] == "General" and "General/Any" in cats):
            score += 2

        # Occupation
        max_score += 2
        occ_tags = str(row["occupation_tags"]).split(",")
        if not row["occupation_tags"] or any(o in occ_tags for o in ans["occupation"]):
            score += 2

        # Disability
        if ans.get("disability"):
            max_score += 1
            if "Person with Disability" in occ_tags:
                score += 1

        return round(100 * score / max_score)

    results = df.copy()
    results["match_score"] = results.apply(score_row, axis=1)

    if ans.get("interests"):
        results = results[results["primary_category"].isin(ans["interests"])]

    results = results.sort_values("match_score", ascending=False)
    top_results = results[results["match_score"] >= 60].head(100)

    st.success(f"✅ Found **{len(top_results)}** schemes matching your profile "
               f"(out of {len(df)} total schemes in our database)")

    hcol1, hcol2 = st.columns([4, 1])
    with hcol2:
        if st.button("🔄 Start Over", use_container_width=True):
            restart()
            st.rerun()

    # Filters on results
    with st.expander("🔧 Refine results"):
        cat_filter = st.multiselect("Category", CATEGORIES, key="refine_cat")
        level_filter = st.multiselect("Level", ["Central", "State"], key="refine_level")
    if cat_filter:
        top_results = top_results[top_results["primary_category"].isin(cat_filter)]
    if level_filter:
        top_results = top_results[top_results["level"].isin(level_filter)]

    if len(top_results) == 0:
        st.warning("No strong matches found. Try adjusting your answers or clearing filters.")

    for _, row in top_results.iterrows():
        level_badge = "badge-central" if row["level"] == "Central" else "badge-state"
        with st.container():
            st.markdown(f"""
            <div class="scheme-card">
                <span class="match-score">{row['match_score']}% match</span>
                <div class="scheme-title">{row['scheme_name']}</div>
                <span class="badge {level_badge}">{row['level']} — {row['state']}</span>
                <span class="badge badge-cat">{row['primary_category']}</span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View details"):
                st.markdown(f"**About:** {row['details'][:600]}{'...' if len(str(row['details']))>600 else ''}")
                st.markdown(f"**Benefits:** {row['benefits'][:500]}{'...' if len(str(row['benefits']))>500 else ''}")
                st.markdown(f"**Eligibility:** {row['eligibility'][:500]}{'...' if len(str(row['eligibility']))>500 else ''}")
                st.markdown(f"**Documents required:** {row['documents'][:400]}{'...' if len(str(row['documents']))>400 else ''}")
                st.caption("⚠️ Always verify current details on the official scheme portal before applying.")

                bcol1, bcol2 = st.columns(2)
                key = row["slug"]
                if key in st.session_state.saved_schemes:
                    if bcol1.button("💔 Remove bookmark", key=f"unsave_{key}"):
                        st.session_state.saved_schemes.discard(key)
                        st.rerun()
                else:
                    if bcol1.button("🔖 Bookmark", key=f"save_{key}"):
                        st.session_state.saved_schemes.add(key)
                        st.rerun()

    st.divider()
    st.caption("Know Your Scheme uses publicly available government scheme data. "
               "Eligibility matching is an estimate to help you discover relevant schemes — "
               "always confirm final eligibility on the official application portal.")

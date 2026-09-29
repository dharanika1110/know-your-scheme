"""
Processes the raw schemes_data_cleaned.csv into a structured CSV with extracted
eligibility fields (state, age range, income ceiling, gender, category, occupation flags)
so the Streamlit app can filter schemes programmatically.

This uses rule-based keyword/regex extraction (no external API calls), so it is
fast, free, and fully reproducible -- but not perfect. Fields that can't be
confidently extracted are left blank, and the app always shows the original
eligibility text too, so nothing is hidden from the user.
"""
import re
import pandas as pd
from states_uts import ALL_STATES_UTS, ALIAS_MAP

IN_PATH = "/mnt/user-data/uploads/schemes_data_cleaned.csv"
OUT_PATH = "/home/claude/knowyourscheme/schemes_processed.csv"

df = pd.read_csv(IN_PATH)

# ---------- STATE EXTRACTION ----------
def extract_state(row):
    if row["level"] == "Central":
        return "All India"
    text = f"{row['scheme_name']} {row['details']} {row['eligibility']}"
    text_l = text.lower()
    found = set()
    for alias_l, canonical in ALIAS_MAP.items():
        # word-boundary-ish match to avoid partial hits
        if re.search(r"\b" + re.escape(alias_l) + r"\b", text_l):
            found.add(canonical)
    if len(found) == 1:
        return list(found)[0]
    elif len(found) > 1:
        # multiple state names mentioned - keep the first mentioned by position
        best, best_pos = None, len(text_l) + 1
        for alias_l, canonical in ALIAS_MAP.items():
            pos = text_l.find(alias_l)
            if pos != -1 and pos < best_pos:
                best, best_pos = canonical, pos
        return best or "Unknown"
    return "Unknown"

df["state"] = df.apply(extract_state, axis=1)

# ---------- AGE EXTRACTION ----------
AGE_RANGE_PATTERNS = [
    r"age\s*group\s*of\s*(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})\s*years",
    r"aged?\s*(?:between)?\s*(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})\s*years",
    r"between\s*(?:the\s*)?ages?\s*of\s*(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})",
]
AGE_MIN_PATTERNS = [
    r"minimum\s*age\s*(?:of|is|:)?\s*(\d{1,2})",
    r"age\s*(?:of\s*)?(?:above|more than|over)\s*(\d{1,2})",
    r"(\d{1,2})\s*years\s*(?:of age\s*)?(?:and above|or above|or more)",
]
AGE_MAX_PATTERNS = [
    r"(?:below|less than|under|up to|not exceeding|maximum age of)\s*(\d{1,2})\s*years",
]

def extract_age(text):
    text_l = text.lower()
    for pat in AGE_RANGE_PATTERNS:
        m = re.search(pat, text_l)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return min(a, b), max(a, b)
    min_age, max_age = None, None
    for pat in AGE_MIN_PATTERNS:
        m = re.search(pat, text_l)
        if m:
            min_age = int(m.group(1))
            break
    for pat in AGE_MAX_PATTERNS:
        m = re.search(pat, text_l)
        if m:
            max_age = int(m.group(1))
            break
    return min_age, max_age

ages = df["eligibility"].fillna("").apply(extract_age)
df["min_age"] = ages.apply(lambda x: x[0])
df["max_age"] = ages.apply(lambda x: x[1])

# ---------- INCOME EXTRACTION ----------
def parse_amount(num_str, unit):
    num_str = num_str.replace(",", "")
    try:
        val = float(num_str)
    except ValueError:
        return None
    unit = (unit or "").lower()
    if "lakh" in unit or "lac" in unit:
        val *= 100000
    elif "crore" in unit:
        val *= 10000000
    return val

INCOME_PATTERNS = [
    r"(?:annual\s+)?(?:family\s+)?income[^.₹]{0,40}?(?:below|less than|up to|not exceed(?:ing)?|under)\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore)?",
    r"₹\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore)?[^.]{0,40}?income",
]

def extract_income(text):
    text_l = text.lower()
    for pat in INCOME_PATTERNS:
        m = re.search(pat, text_l)
        if m:
            val = parse_amount(m.group(1), m.group(2))
            if val:
                return val
    return None

df["max_income"] = df["eligibility"].fillna("").apply(extract_income)

# ---------- GENDER ----------
def extract_gender(text):
    text_l = text.lower()
    female_kw = ["woman", "women", "female", "girl", "widow", "mother"]
    male_kw = ["male applicant", "only male", "men only"]
    has_female = any(k in text_l for k in female_kw)
    has_male = any(k in text_l for k in male_kw)
    if has_female and not has_male:
        return "Female"
    if has_male and not has_female:
        return "Male"
    return "Any"

df["gender"] = df["eligibility"].fillna("").apply(extract_gender)

# ---------- CATEGORY (caste) ----------
def extract_caste(text):
    text_l = text.lower()
    tags = []
    if re.search(r"\bsc\b|scheduled caste", text_l):
        tags.append("SC")
    if re.search(r"\bst\b|scheduled tribe", text_l):
        tags.append("ST")
    if re.search(r"\bobc\b|other backward class", text_l):
        tags.append("OBC")
    if re.search(r"\bews\b|economically weaker section", text_l):
        tags.append("EWS")
    if re.search(r"minorit", text_l):
        tags.append("Minority")
    return ",".join(tags) if tags else "General/Any"

df["caste_category"] = df["eligibility"].fillna("").apply(extract_caste)

# ---------- OCCUPATION / LIFE-STAGE FLAGS ----------
OCCUPATION_KEYWORDS = {
    "Student": ["student", "school", "college", "scholarship", "education"],
    "Farmer": ["farmer", "agricultur", "cultivat", "crop", "land holder"],
    "Fisherman": ["fisherman", "fisherwoman", "fishermen", "fishing"],
    "Laborer/Worker": ["labour", "labor", "worker", "construction worker", "unorganised worker"],
    "Entrepreneur/MSME": ["entrepreneur", "msme", "startup", "self-employ", "business"],
    "Senior Citizen": ["senior citizen", "old age", "elderly"],
    "Person with Disability": ["disability", "disabled", "divyang", "pwd"],
    "Unemployed": ["unemployed", "job seeker"],
    "Artisan/Craftsman": ["artisan", "weaver", "handicraft", "handloom"],
    "Widow": ["widow"],
}

def extract_occupations(text):
    text_l = text.lower()
    hits = [name for name, kws in OCCUPATION_KEYWORDS.items() if any(k in text_l for k in kws)]
    return ",".join(hits) if hits else ""

df["occupation_tags"] = (df["eligibility"].fillna("") + " " + df["tags"].fillna("")).apply(extract_occupations)

# ---------- CLEAN CATEGORY (scheme domain, not eligibility) ----------
df["primary_category"] = df["schemeCategory"].fillna("Other").apply(lambda x: x.split(",")[0].strip())

# ---------- SAVE ----------
cols = [
    "scheme_name", "slug", "level", "state", "primary_category", "schemeCategory",
    "min_age", "max_age", "max_income", "gender", "caste_category", "occupation_tags",
    "details", "benefits", "eligibility", "application", "documents", "tags",
]
df[cols].to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Rows:", len(df))
print("\nState coverage:")
print(df["state"].value_counts().head(15))
print("\nAge extracted (non-null min_age):", df["min_age"].notna().sum())
print("Income extracted (non-null):", df["max_income"].notna().sum())
print("Gender != Any:", (df["gender"] != "Any").sum())
print("Caste tagged:", (df["caste_category"] != "General/Any").sum())
print("Occupation tagged:", (df["occupation_tags"] != "").sum())

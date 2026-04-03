import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import pandas as pd
import math
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    model  = joblib.load(os.path.join(BASE_DIR, "score_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler

model, scaler = load_models()

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Farmer Credit Score",
    page_icon="🌾",
    layout="centered"
)

st.markdown("""
<style>
/* ── Advanced GitHub-Dark Farmer Theme ── */
.stApp { background-color: #0d1117 !important; color: #e6edf3 !important; }
section.main > div { background-color: #0d1117; }

h1,h2,h3 { color: #3fb950 !important; font-weight: 500 !important; }
.stSubheader { color: #3fb950 !important; }
label, .stMarkdown p { color: #8b949e !important; }
.stCaption { color: #484f58 !important; }
hr { border-color: #2d333b !important; }

/* Inputs */
input[type="number"], input[type="text"] {
    background: #1c2128 !important; color: #e6edf3 !important;
    border: 0.5px solid #2d333b !important; border-radius: 8px !important;
}
input:focus { border-color: #238636 !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #1c2128 !important; color: #e6edf3 !important;
    border: 0.5px solid #2d333b !important; border-radius: 8px !important;
}

/* Button */
.stButton > button[kind="primary"] {
    background: #238636 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-size: 15px !important; font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s !important;
}
.stButton > button[kind="primary"]:hover { background: #3fb950 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22 !important; border: 0.5px solid #2d333b !important;
    border-radius: 10px !important; padding: 14px !important;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }

/* Alert / info boxes */
.stAlert {
    background: #1c2128 !important; border: 0.5px solid #2d333b !important;
    border-radius: 8px !important; color: #e6edf3 !important;
}
div[data-baseweb="notification"] { background: #1a4731 !important; }

/* Success = green tint */
.stSuccess { border-left: 3px solid #3fb950 !important; background: #0d2016 !important; }
/* Warning = amber tint */
.stWarning { border-left: 3px solid #e3b341 !important; background: #1f1a0a !important; }
/* Error = red tint */
.stError { border-left: 3px solid #f85149 !important; background: #200d0d !important; }

/* Charts */
[data-testid="stVegaLiteChart"] {
    background: #161b22 !important; border: 0.5px solid #2d333b !important;
    border-radius: 10px !important; padding: 10px !important;
}

/* Slider */
.stSlider > div > div > div { background: #238636 !important; }

/* Columns gap */
[data-testid="column"] { gap: 8px; }

/* Map iframe */
iframe { border-radius: 10px !important; border: 0.5px solid #2d333b !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important; border-right: 0.5px solid #2d333b !important;
}

/* Number input up/down arrows */
button[data-baseweb="button"] {
    background: #1c2128 !important; color: #8b949e !important;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
#  TRANSLATIONS
# ─────────────────────────────────────────────
T = {
    "Tamil": {
        "title":            "🌾 விவசாயி கடன் மதிப்பீடு",
        "subtitle":         "உங்கள் Credit Score கணக்கிட்டு, அருகிலுள்ள வங்கிகளை கண்டறியுங்கள்",
        # form sections
        "basic_info":       "📋 அடிப்படை தகவல்",
        "farm_info":        "🌾 விவசாய விவரம்",
        "personal_info":    "👤 தனிப்பட்ட தகவல்",
        # fields
        "land":             "நில அளவு (ஏக்கர்)",
        "crop":             "பயிர் வகை",
        "exp":              "அனுபவம் (ஆண்டுகள்)",
        "loan":             "கடன் வரலாறு",
        "annual_income":    "ஆண்டு வருமானம் (₹)",
        "land_type":        "நில வகை",
        "soil_quality":     "மண் தரம்",
        "rainfall":         "மழை அளவு (மிமீ/ஆண்டு)",
        "crop_yield":       "பயிர் விளைச்சல் (குவிண்டால்/ஏக்கர்)",
        "water_source":     "நீர் ஆதாரம்",
        "age":              "வயது",
        "education":        "கல்வி தகுதி",
        "existing_loans":   "தற்போதுள்ள கடன்கள் (எண்ணிக்கை)",
        "repayment_delay":  "கடன் தாமத நாட்கள் (கடந்த கடனில்)",
        # options
        "crops":            ["நெல்", "கரும்பு", "வாழை", "பருத்தி", "தக்காளி"],
        "loans":            ["திருப்பினேன்", "கடன் இல்லை", "பகுதியாக திருப்பினேன்", "திருப்பவில்லை"],
        "land_type_opts":   ["நீர்பாசனம்", "மானாவாரி"],
        "soil_quality_opts":["மோசம்", "சராசரி", "நல்லது"],
        "water_source_opts":["கால்வாய்", "கிணறு / போர்வெல்", "மழைநீர்"],
        "education_opts":   ["படிக்காதவர்", "தொடக்கப்பள்ளி", "உயர்நிலை", "கல்லூரி"],
        # results
        "btn":              "Credit Score கணக்கிடு",
        "result":           "உங்கள் Credit Score முடிவு",
        "eligible":         "✅ தகுதியுள்ளவர்",
        "conditional":      "⚠️ நிபந்தனையுடன்",
        "ineligible":       "❌ தகுதியற்றவர்",
        "max_loan":         "அதிகபட்ச கடன் தொகை",
        "interest":         "வட்டி விகிதம்",
        "schemes":          "🏛️ அரசு திட்டங்கள்",
        "breakdown":        "Score விவரம்",
        "nearby":           "🏦 அருகிலுள்ள வங்கிகள்",
        "district":         "உங்கள் மாவட்டம் தேர்வு செய்யுங்கள்",
        "contact":          "📋 வங்கி தொடர்பு எண்கள்",
        "locate_tip":       "📍 உங்கள் இடத்தை map-ல் காட்ட மேல் வலதுபுறம் உள்ள location button கிளிக் செய்யுங்கள்!",
    },
    "Hindi": {
        "title":            "🌾 किसान क्रेडिट स्कोर",
        "subtitle":         "अपना क्रेडिट स्कोर जानें और नजदीकी बैंक खोजें",
        "basic_info":       "📋 बुनियादी जानकारी",
        "farm_info":        "🌾 खेत की जानकारी",
        "personal_info":    "👤 व्यक्तिगत जानकारी",
        "land":             "जमीन (एकड़)",
        "crop":             "फसल",
        "exp":              "अनुभव (वर्ष)",
        "loan":             "ऋण इतिहास",
        "annual_income":    "वार्षिक आय (₹)",
        "land_type":        "भूमि प्रकार",
        "soil_quality":     "मिट्टी की गुणवत्ता",
        "rainfall":         "वर्षा (मिमी/वर्ष)",
        "crop_yield":       "फसल उत्पादन (क्विंटल/एकड़)",
        "water_source":     "जल स्रोत",
        "age":              "आयु",
        "education":        "शिक्षा स्तर",
        "existing_loans":   "मौजूदा ऋण (संख्या)",
        "repayment_delay":  "भुगतान में देरी (दिन)",
        "crops":            ["धान", "गन्ना", "केला", "कपास", "टमाटर"],
        "loans":            ["चुकाया", "कोई ऋण नहीं", "आंशिक चुकाया", "नहीं चुकाया"],
        "land_type_opts":   ["सिंचित", "वर्षा आधारित"],
        "soil_quality_opts":["खराब", "सामान्य", "अच्छी"],
        "water_source_opts":["नहर", "कुआं / बोरवेल", "वर्षा जल"],
        "education_opts":   ["अशिक्षित", "प्राथमिक", "माध्यमिक", "स्नातक"],
        "btn":              "क्रेडिट स्कोर देखें",
        "result":           "आपका क्रेडिट स्कोर परिणाम",
        "eligible":         "✅ पात्र",
        "conditional":      "⚠️ सशर्त",
        "ineligible":       "❌ अपात्र",
        "max_loan":         "अधिकतम ऋण राशि",
        "interest":         "ब्याज दर",
        "schemes":          "🏛️ सरकारी योजनाएं",
        "breakdown":        "स्कोर विवरण",
        "nearby":           "🏦 नजदीकी बैंक",
        "district":         "अपना जिला चुनें",
        "contact":          "📋 बैंक संपर्क नंबर",
        "locate_tip":       "📍 अपनी लाइव लोकेशन देखने के लिए मैप पर ऊपर दाईं ओर location button दबाएं!",
    },
    "English": {
        "title":            "🌾 Farmer Credit Score",
        "subtitle":         "Calculate your credit score and find nearby banks",
        "basic_info":       "📋 Basic Information",
        "farm_info":        "🌾 Farm Details",
        "personal_info":    "👤 Personal Details",
        "land":             "Land area (acres)",
        "crop":             "Crop type",
        "exp":              "Experience (years)",
        "loan":             "Loan history",
        "annual_income":    "Annual income (₹)",
        "land_type":        "Land type",
        "soil_quality":     "Soil quality",
        "rainfall":         "Rainfall (mm/year)",
        "crop_yield":       "Crop yield (quintal/acre)",
        "water_source":     "Water source",
        "age":              "Age",
        "education":        "Education level",
        "existing_loans":   "Existing loans (count)",
        "repayment_delay":  "Repayment delay (days, last loan)",
        "crops":            ["Paddy", "Sugarcane", "Banana", "Cotton", "Tomato"],
        "loans":            ["Repaid", "No loan", "Partially repaid", "Not repaid"],
        "land_type_opts":   ["Irrigated", "Rain-fed"],
        "soil_quality_opts":["Poor", "Average", "Good"],
        "water_source_opts":["Canal", "Well / Borewell", "Rainwater"],
        "education_opts":   ["Illiterate", "Primary", "Secondary", "Graduate"],
        "btn":              "Calculate Credit Score",
        "result":           "Your Credit Score Result",
        "eligible":         "✅ Eligible",
        "conditional":      "⚠️ Conditional",
        "ineligible":       "❌ Not Eligible",
        "max_loan":         "Max loan amount",
        "interest":         "Interest rate",
        "schemes":          "🏛️ Government Schemes",
        "breakdown":        "Score Breakdown",
        "nearby":           "🏦 Nearby Banks",
        "district":         "Select your district",
        "contact":          "📋 Bank Contact Numbers",
        "locate_tip":       "📍 Click the location button at top-right on the map to show your live GPS location!",
    }
}

crop_scores = [90, 90, 75, 75, 65]
loan_scores  = [100, 75, 45, 10]

# ─────────────────────────────────────────────
#  BANK DATA
# ─────────────────────────────────────────────
ALL_BANKS = [
    {"name": "SBI Chennai Main",         "lat": 13.0839, "lon": 80.2785, "phone": "1800-11-2211",  "color": "blue",   "district": "Chennai"},
    {"name": "Indian Bank Anna Salai",   "lat": 13.0672, "lon": 80.2518, "phone": "1800-425-0000", "color": "green",  "district": "Chennai"},
    {"name": "Canara Bank T Nagar",      "lat": 13.0418, "lon": 80.2341, "phone": "1800-103-0018", "color": "orange", "district": "Chennai"},
    {"name": "NABARD Chennai",           "lat": 13.0100, "lon": 80.2706, "phone": "044-28313000",  "color": "red",    "district": "Chennai"},
    {"name": "SBI Coimbatore Main",      "lat": 11.0168, "lon": 76.9680, "phone": "1800-11-2211",  "color": "blue",   "district": "Coimbatore"},
    {"name": "Indian Bank RS Puram",     "lat": 11.0050, "lon": 76.9550, "phone": "1800-425-0000", "color": "green",  "district": "Coimbatore"},
    {"name": "Canara Bank Gandhipuram",  "lat": 11.0200, "lon": 76.9750, "phone": "1800-103-0018", "color": "orange", "district": "Coimbatore"},
    {"name": "NABARD Coimbatore",        "lat": 11.0300, "lon": 76.9600, "phone": "0422-2300491",  "color": "red",    "district": "Coimbatore"},
    {"name": "SBI Madurai Main",         "lat":  9.9195, "lon": 78.1193, "phone": "1800-11-2211",  "color": "blue",   "district": "Madurai"},
    {"name": "Indian Bank Madurai",      "lat":  9.9300, "lon": 78.1250, "phone": "1800-425-0000", "color": "green",  "district": "Madurai"},
    {"name": "Canara Bank Madurai",      "lat":  9.9100, "lon": 78.1100, "phone": "1800-103-0018", "color": "orange", "district": "Madurai"},
    {"name": "NABARD Madurai",           "lat":  9.9400, "lon": 78.1300, "phone": "0452-2530844",  "color": "red",    "district": "Madurai"},
    {"name": "SBI Thanjavur Main",       "lat": 10.7870, "lon": 79.1450, "phone": "1800-11-2211",  "color": "blue",   "district": "Thanjavur"},
    {"name": "Indian Bank Thanjavur",    "lat": 10.7920, "lon": 79.1500, "phone": "1800-425-0000", "color": "green",  "district": "Thanjavur"},
    {"name": "Canara Bank Thanjavur",    "lat": 10.7800, "lon": 79.1350, "phone": "1800-103-0018", "color": "orange", "district": "Thanjavur"},
    {"name": "NABARD Thanjavur",         "lat": 10.7950, "lon": 79.1400, "phone": "04362-230844",  "color": "red",    "district": "Thanjavur"},
    {"name": "SBI Salem Main",           "lat": 11.6700, "lon": 78.1500, "phone": "1800-11-2211",  "color": "blue",   "district": "Salem"},
    {"name": "Indian Bank Salem",        "lat": 11.6600, "lon": 78.1400, "phone": "1800-425-0000", "color": "green",  "district": "Salem"},
    {"name": "Canara Bank Salem",        "lat": 11.6800, "lon": 78.1600, "phone": "1800-103-0018", "color": "orange", "district": "Salem"},
    {"name": "NABARD Salem",             "lat": 11.6500, "lon": 78.1300, "phone": "0427-2230844",  "color": "red",    "district": "Salem"},
    {"name": "SBI Trichy Main",          "lat": 10.7950, "lon": 78.7100, "phone": "1800-11-2211",  "color": "blue",   "district": "Trichy"},
    {"name": "Indian Bank Trichy",       "lat": 10.7850, "lon": 78.7000, "phone": "1800-425-0000", "color": "green",  "district": "Trichy"},
    {"name": "Canara Bank Trichy",       "lat": 10.8000, "lon": 78.7200, "phone": "1800-103-0018", "color": "orange", "district": "Trichy"},
    {"name": "NABARD Trichy",            "lat": 10.7750, "lon": 78.6900, "phone": "0431-2460844",  "color": "red",    "district": "Trichy"},
    {"name": "SBI Erode Main",           "lat": 11.3450, "lon": 77.7200, "phone": "1800-11-2211",  "color": "blue",   "district": "Erode"},
    {"name": "Indian Bank Erode",        "lat": 11.3350, "lon": 77.7100, "phone": "1800-425-0000", "color": "green",  "district": "Erode"},
    {"name": "Canara Bank Erode",        "lat": 11.3500, "lon": 77.7300, "phone": "1800-103-0018", "color": "orange", "district": "Erode"},
    {"name": "NABARD Erode",             "lat": 11.3250, "lon": 77.7000, "phone": "0424-2230844",  "color": "red",    "district": "Erode"},
    {"name": "SBI Tirunelveli Main",     "lat":  8.7200, "lon": 77.7600, "phone": "1800-11-2211",  "color": "blue",   "district": "Tirunelveli"},
    {"name": "Indian Bank Tirunelveli",  "lat":  8.7100, "lon": 77.7500, "phone": "1800-425-0000", "color": "green",  "district": "Tirunelveli"},
    {"name": "Canara Bank Tirunelveli",  "lat":  8.7300, "lon": 77.7700, "phone": "1800-103-0018", "color": "orange", "district": "Tirunelveli"},
    {"name": "NABARD Tirunelveli",       "lat":  8.7000, "lon": 77.7400, "phone": "0462-2330844",  "color": "red",    "district": "Tirunelveli"},
]

DISTRICT_COORDS = {
    "Chennai":     [13.0827, 80.2707],
    "Coimbatore":  [11.0168, 76.9558],
    "Madurai":     [ 9.9252, 78.1198],
    "Thanjavur":   [10.7870, 79.1378],
    "Salem":       [11.6643, 78.1460],
    "Trichy":      [10.7905, 78.7047],
    "Erode":       [11.3410, 77.7172],
    "Tirunelveli": [ 8.7139, 77.7567],
}

# ─────────────────────────────────────────────
#  HAVERSINE
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return round(R * 2 * math.asin(math.sqrt(a)), 1)

def get_nearby_banks(user_lat, user_lon, top_n=5):
    ranked = [{**b, "distance_km": haversine(user_lat, user_lon, b["lat"], b["lon"])} for b in ALL_BANKS]
    ranked.sort(key=lambda x: x["distance_km"])
    return ranked[:top_n]

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in [("calculated", False), ("result_data", {}), ("district", "Coimbatore")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
#  LANGUAGE
# ─────────────────────────────────────────────
lang = st.selectbox("🌐 Language / மொழி / भाषा", ["Tamil", "Hindi", "English"])
t = T[lang]

st.title(t["title"])
st.caption(t["subtitle"])
st.markdown("---")

# ─────────────────────────────────────────────
#  FORM — Section 1: Basic Info
# ─────────────────────────────────────────────
st.subheader(t["basic_info"])
c1, c2 = st.columns(2)
with c1:
    land     = st.number_input(t["land"], min_value=0.5, max_value=100.0, value=5.0, step=0.5)
    crop_idx = st.selectbox(t["crop"], range(len(t["crops"])), format_func=lambda x: t["crops"][x])
with c2:
    exp      = st.number_input(t["exp"], min_value=1, max_value=50, value=10)
    loan_idx = st.selectbox(t["loan"], range(len(t["loans"])), format_func=lambda x: t["loans"][x])

st.markdown("")

# — Section 2: Farm Details —
st.subheader(t["farm_info"])
c3, c4 = st.columns(2)
with c3:
    land_type_idx    = st.selectbox(t["land_type"],    range(len(t["land_type_opts"])),    format_func=lambda x: t["land_type_opts"][x])
    soil_quality_idx = st.selectbox(t["soil_quality"], range(len(t["soil_quality_opts"])), format_func=lambda x: t["soil_quality_opts"][x])
    water_source_idx = st.selectbox(t["water_source"], range(len(t["water_source_opts"])), format_func=lambda x: t["water_source_opts"][x])
with c4:
    rainfall     = st.number_input(t["rainfall"],  min_value=100,  max_value=3000,    value=800,    step=50)
    crop_yield   = st.number_input(t["crop_yield"],min_value=1,    max_value=200,     value=50,     step=1)
    annual_income = st.number_input(t["annual_income"], min_value=10000, max_value=5000000, value=200000, step=10000)

st.markdown("")

# — Section 3: Personal Details —
st.subheader(t["personal_info"])
c5, c6 = st.columns(2)
with c5:
    age           = st.number_input(t["age"],            min_value=18, max_value=80, value=35)
    education_idx = st.selectbox(t["education"], range(len(t["education_opts"])), format_func=lambda x: t["education_opts"][x])
with c6:
    existing_loans   = st.number_input(t["existing_loans"],  min_value=0,  max_value=10,  value=0,  step=1)
    repayment_delay  = st.number_input(t["repayment_delay"], min_value=0,  max_value=365, value=0,  step=1)

st.markdown("")

# ─────────────────────────────────────────────
#  CALCULATE BUTTON
# ─────────────────────────────────────────────
if st.button(t["btn"], use_container_width=True, type="primary"):

    # Encode categorical → numeric (1-indexed to match training)
    crop_val         = crop_idx + 1
    loan_val         = loan_idx
    land_type_val    = land_type_idx + 1   # 1=Irrigated, 2=Rain-fed
    soil_val         = soil_quality_idx + 1  # 1=Poor, 2=Average, 3=Good
    water_source_val = water_source_idx + 1  # 1=Canal, 2=Well/Borewell, 3=Rainwater
    edu_val          = education_idx + 1   # 1–4

    # Build input — all 14 features, all from user, zero hardcoded
    input_df = pd.DataFrame([{
        "land":            land,
        "experience":      exp,
        "loan_history":    loan_val,
        "crop":            crop_val,
        "annual_income":   annual_income,
        "existing_loans":  existing_loans,
        "repayment_delay": repayment_delay,
        "land_type":       land_type_val,
        "crop_yield":      crop_yield,
        "soil_quality":    soil_val,
        "rainfall":        rainfall,
        "water_source":    water_source_val,
        "age":             age,
        "education_level": edu_val,
    }])

    input_scaled  = scaler.transform(input_df)
    total         = int(model.predict(input_scaled)[0])

    importance_df = pd.DataFrame({
        "Feature":    input_df.columns,
        "Importance": model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    st.session_state.calculated  = True
    st.session_state.result_data = {
        "total":         total,
        "land":          land,
        "lang":          lang,
        "importance_df": importance_df,
    }

# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
if st.session_state.calculated:
    d     = st.session_state.result_data
    total = d["total"]
    land  = d["land"]
    t2    = T[d["lang"]]

    st.markdown("---")
    st.subheader(t2["result"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{total} / 600")

    if total >= 450:
        c2.success(t2["eligible"])
        c3.metric(t2["interest"], "7% p.a.")
        loan_amt = round(land * 0.5, 1)
    elif total >= 350:
        c2.warning(t2["conditional"])
        c3.metric(t2["interest"], "9% p.a.")
        loan_amt = round(land * 0.3, 1)
    else:
        c2.error(t2["ineligible"])
        c3.metric(t2["interest"], "—")
        loan_amt = 0

    if loan_amt > 0:
        st.metric(t2["max_loan"], f"₹ {loan_amt} Lakh")
    else:
        st.info("Loan not eligible at this time. Improve your score and try again.")

    # Score breakdown
    st.markdown("---")
    st.subheader(t2["breakdown"])
    st.bar_chart(d["importance_df"].set_index("Feature"))

    # Government schemes
    st.markdown("---")
    st.subheader(t2["schemes"])
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.info("**KCC Loan**\nKisan Credit Card — easy farm credit at low interest rate.")
    with sc2:
        st.info("**PM-KISAN**\n₹6,000/year deposited directly to farmer's bank account.")
    with sc3:
        st.info("**PMFBY**\nCrop insurance scheme covering all major crops.")

    # ─────────────────────────────────────────
    #  NEARBY BANKS
    # ─────────────────────────────────────────
    st.markdown("---")
    st.subheader(t2["nearby"])
    st.info(t2["locate_tip"])

    selected_district = st.selectbox(
        t2["district"],
        list(DISTRICT_COORDS.keys()),
        index=list(DISTRICT_COORDS.keys()).index(st.session_state.district),
    )
    st.session_state.district = selected_district

    user_lat, user_lon = DISTRICT_COORDS[selected_district]
    nearby_banks = get_nearby_banks(user_lat, user_lon, top_n=5)

    m = folium.Map(location=[user_lat, user_lon], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        location=[user_lat, user_lon],
        tooltip="📍 உங்கள் இடம் / Your Location",
        popup="You are here",
        icon=folium.Icon(color="purple", icon="user", prefix="fa"),
    ).add_to(m)

    for bank in nearby_banks:
        popup_html = f"""
        <div style="font-family:Arial;width:220px;padding:6px">
            <h4 style="color:#1a73e8;margin:0 0 6px 0">{bank['name']}</h4>
            <hr style="margin:4px 0">
            <p style="margin:4px 0">📞 <b>{bank['phone']}</b></p>
            <p style="margin:4px 0">📍 <b>{bank['distance_km']} km</b> away</p>
            <p style="margin:4px 0">✅ KCC Loan available</p>
            <p style="margin:4px 0">🕐 Mon–Sat: 10 AM – 4 PM</p>
            <a href="tel:{bank['phone']}"
               style="background:#1a73e8;color:#fff;padding:5px 12px;
                      border-radius:4px;text-decoration:none;
                      display:inline-block;margin-top:6px">
               📞 Call Now
            </a>
        </div>
        """
        folium.Marker(
            location=[bank["lat"], bank["lon"]],
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"🏦 {bank['name']}  |  📍 {bank['distance_km']} km  |  📞 {bank['phone']}",
            icon=folium.Icon(color=bank["color"], icon="home"),
        ).add_to(m)
        folium.PolyLine(
            locations=[[user_lat, user_lon], [bank["lat"], bank["lon"]]],
            color="#1a73e8", weight=1.5, opacity=0.4, dash_array="6 4",
        ).add_to(m)

    plugins.LocateControl(auto_start=False, position="topright").add_to(m)
    st_folium(m, width=700, height=480, key=f"map_{selected_district}")

    st.markdown(f"### {t2['contact']}")
    st.caption("Sorted by distance — nearest bank first 🏦")
    for i, bank in enumerate(nearby_banks, 1):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.write(f"**{i}. {bank['name']}** ({bank['district']})")
        with c2:
            st.write(f"📍 {bank['distance_km']} km")
        with c3:
            st.write(f"📞 {bank['phone']}")

    st.markdown("---")
    st.caption("🌾 Developed for Tamil Nadu Farmers  |  Data is indicative only")

    # ─────────────────────────────────────────
    #  EMI CALCULATOR
    # ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 EMI Calculator")

    if loan_amt > 0:
        default_interest = 7.0 if total >= 700 else (9.0 if total >= 550 else 10.0)

        c1, c2 = st.columns(2)
        with c1:
            principal     = st.number_input("Loan Amount (₹ Lakh)", min_value=0.1, value=float(loan_amt))
            interest_rate = st.slider("Interest Rate (%)", 5.0, 15.0, default_interest)
        with c2:
            tenure_years = st.slider("Loan Tenure (Years)", 1, 10, 5)

        P = principal * 100000
        R = interest_rate / (12 * 100)
        N = tenure_years * 12
        emi = (P * R * (1 + R) ** N) / ((1 + R) ** N - 1) if R > 0 else P / N

        st.success(f"📌 Monthly EMI: ₹ {int(emi):,}")
        st.info(f"💵 Total Payment: ₹ {int(emi * N):,}")
        st.info(f"📈 Total Interest: ₹ {int(emi * N - P):,}")
    else:
        st.warning("EMI Calculator available only for eligible loans.")
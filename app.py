"""
IPL Win Probability Predictor - Streamlit Web Application
==========================================================
Interactive Web Application for predicting ball-by-ball win probabilities
in Indian Premier League (IPL) second innings run chases using Machine Learning.

Author: College NTCC Project & Portfolio
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Main container and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Rajdhani:wght@600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #F59E0B 0%, #EF4444 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }

    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(245, 158, 11, 0.4);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-val {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-val.highlight {
        color: #F59E0B;
    }

    /* Result Probability Cards */
    .prob-card {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 10px;
    }
    .prob-card-batting {
        background: linear-gradient(145deg, #1E3A8A 0%, #172554 100%);
        border-top: 4px solid #3B82F6;
    }
    .prob-card-bowling {
        background: linear-gradient(145deg, #831843 0%, #500724 100%);
        border-top: 4px solid #EC4899;
    }
    .team-name {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .prob-pct {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3.4rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
        margin-bottom: 12px;
    }

    /* Banner badge */
    .verdict-banner {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(239, 68, 68, 0.15) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 14px 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.15rem;
        font-weight: 600;
        color: #FDE68A;
        margin: 20px 0;
    }

    /* Clean inputs container */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. Constants & Data Defaults
# ==========================================
ACTIVE_TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

COMMON_CITIES = [
    'Ahmedabad', 'Bangalore', 'Chandigarh', 'Chennai', 'Cuttack',
    'Delhi', 'Dharamsala', 'Dubai', 'Hyderabad', 'Indore',
    'Jaipur', 'Kolkata', 'Lucknow', 'Mohali', 'Mumbai',
    'Navi Mumbai', 'Pune', 'Raipur', 'Ranchi', 'Sharjah', 'Visakhapatnam'
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pipe.pkl')
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'metadata.json')


# ==========================================
# 3. Model Loading Helper
# ==========================================
@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline object."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            pipe = pickle.load(f)
        return pipe
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None

pipe = load_model()


# ==========================================
# 4. Sidebar Presets & NTCC Metadata
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1200px-Indian_Premier_League_Official_Logo.svg.png", width=140)
    st.markdown("### 🏆 IPL Analytics Engine")
    st.markdown("Predict dynamic second-innings victory probability based on historical chase conditions.")
    
    st.divider()
    st.subheader("⚡ Quick Match Scenarios")
    scenario = st.selectbox(
        "Load Preset Chase Situation:",
        [
            "Custom User Input",
            "Nail-Biter (12 off 6 balls, 6 wkts)",
            "Commanding Chase (45 off 48 balls, 8 wkts)",
            "Top Order Collapse (130 off 60 balls, 3 wkts)",
            "Powerplay Rampage (65/1 in 6.0 overs, Target 185)",
            "High Pressure (75 off 36 balls, 5 wkts)"
        ]
    )

    # Preset logic
    defaults = {
        'batting': 'Royal Challengers Bangalore',
        'bowling': 'Chennai Super Kings',
        'city': 'Bangalore',
        'target': 180,
        'score': 95,
        'overs': 11.4,
        'wickets': 3
    }

    if scenario == "Nail-Biter (12 off 6 balls, 6 wkts)":
        defaults = {'batting': 'Mumbai Indians', 'bowling': 'Chennai Super Kings', 'city': 'Mumbai', 'target': 190, 'score': 178, 'overs': 19.0, 'wickets': 4}
    elif scenario == "Commanding Chase (45 off 48 balls, 8 wkts)":
        defaults = {'batting': 'Kolkata Knight Riders', 'bowling': 'Rajasthan Royals', 'city': 'Kolkata', 'target': 160, 'score': 115, 'overs': 12.0, 'wickets': 2}
    elif scenario == "Top Order Collapse (130 off 60 balls, 3 wkts)":
        defaults = {'batting': 'Delhi Capitals', 'bowling': 'Sunrisers Hyderabad', 'city': 'Delhi', 'target': 205, 'score': 75, 'overs': 10.0, 'wickets': 7}
    elif scenario == "Powerplay Rampage (65/1 in 6.0 overs, Target 185)":
        defaults = {'batting': 'Sunrisers Hyderabad', 'bowling': 'Punjab Kings', 'city': 'Hyderabad', 'target': 185, 'score': 65, 'overs': 6.0, 'wickets': 1}
    elif scenario == "High Pressure (75 off 36 balls, 5 wkts)":
        defaults = {'batting': 'Gujarat Titans', 'bowling': 'Lucknow Super Giants', 'city': 'Ahmedabad', 'target': 195, 'score': 120, 'overs': 14.0, 'wickets': 5}

    st.divider()
    st.markdown("### 🎓 NTCC Project Info")
    st.markdown("""
    - **Domain:** Sports Analytics & Predictive ML
    - **Model:** Logistic Regression with OneHotEncoding
    - **Metrics:** CRR, RRR, Wickets in hand, Venue bias
    """)


# ==========================================
# 5. Main UI Header
# ==========================================
st.markdown('<div class="main-title">🏏 IPL WIN PROBABILITY PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Machine Learning Win Forecaster for 2nd Innings Run Chases</div>', unsafe_allow_html=True)

# Missing model alert & helper
if pipe is None:
    st.warning("⚠️ **Trained model pipeline not found in `models/pipe.pkl`.**")
    st.info("Run `python train_model.py` in your terminal to train and serialize the model. If you don't have Kaggle datasets yet, the training script will automatically create realistic sample data and train immediately!")
    if st.button("🚀 Auto-Train Model Now (from App)", type="primary"):
        with st.spinner("Training model pipeline..."):
            try:
                import train_model
                data = train_model.load_and_preprocess_data()
                pipe, _ = train_model.train_and_evaluate_model(data)
                st.success("✅ Model trained successfully! Please reload the page.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

# ==========================================
# 6. User Input Sections
# ==========================================
with st.container():
    st.subheader("1. Match Setup")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        batting_team = st.selectbox(
            "🏏 Chasing / Batting Team",
            ACTIVE_TEAMS,
            index=ACTIVE_TEAMS.index(defaults['batting']) if defaults['batting'] in ACTIVE_TEAMS else 0
        )
    with col2:
        # Filter bowling teams to prevent picking the same team
        bowling_options = [team for team in ACTIVE_TEAMS if team != batting_team]
        bowling_index = bowling_options.index(defaults['bowling']) if defaults['bowling'] in bowling_options else 0
        bowling_team = st.selectbox("🎯 Defending / Bowling Team", bowling_options, index=bowling_index)

    with col3:
        host_city = st.selectbox(
            "📍 Host City / Venue",
            COMMON_CITIES,
            index=COMMON_CITIES.index(defaults['city']) if defaults['city'] in COMMON_CITIES else 0
        )

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    st.subheader("2. Current Chase Situation")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        target_score = st.number_input(
            "🎯 Target Score",
            min_value=50,
            max_value=320,
            value=int(defaults['target']),
            step=1,
            help="Total runs required to win the match (1st Innings Score + 1)"
        )
    with c2:
        current_score = st.number_input(
            "📊 Current Score",
            min_value=0,
            max_value=320,
            value=int(defaults['score']),
            step=1,
            help="Runs scored so far by the chasing team"
        )
    with c3:
        overs_completed = st.number_input(
            "⏱️ Overs Completed (e.g. 10.2)",
            min_value=0.0,
            max_value=20.0,
            value=float(defaults['overs']),
            step=0.1,
            help="Overs bowled. Decimal indicates balls (e.g., 10.2 means 10 overs and 2 balls, balls must be 0 to 5)"
        )
    with c4:
        wickets_lost = st.number_input(
            "❌ Wickets Lost",
            min_value=0,
            max_value=10,
            value=int(defaults['wickets']),
            step=1,
            help="Number of wickets fallen (0 to 10)"
        )


# ==========================================
# 7. Overs & Cricket Calculations
# ==========================================
# Parse overs accurately: e.g. 10.2 -> 10 overs + 2 balls = 62 balls
full_overs = int(overs_completed)
extra_balls = int(round((overs_completed - full_overs) * 10))

# Validate illegal cricket balls (e.g. 10.6, 10.7)
invalid_overs = False
if extra_balls >= 6:
    st.error(f"⚠️ Invalid cricket over format `{overs_completed}`! In cricket, an over has at most 5 extra balls (e.g., 10.0 to 10.5). Next ball is 11.0.")
    invalid_overs = True
    extra_balls = 5

total_balls_bowled = (full_overs * 6) + extra_balls
balls_left = max(0, 120 - total_balls_bowled)
runs_left = target_score - current_score
wickets_left = max(0, 10 - wickets_lost)

# Calculate Run Rates
crr = (current_score * 6) / total_balls_bowled if total_balls_bowled > 0 else 0.0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 8. Dynamic Match KPI Cards
# ==========================================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Runs Needed</div>
        <div class="metric-val highlight">{max(0, runs_left)}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Balls Left</div>
        <div class="metric-val">{balls_left}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Wickets in Hand</div>
        <div class="metric-val">{wickets_left}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Run Rate</div>
        <div class="metric-val">{crr:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Required Run Rate</div>
        <div class="metric-val highlight">{rrr:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# 9. Prediction Logic & Visual Results
# ==========================================
predict_btn = st.button("⚡ Predict Win Probability", type="primary", use_container_width=True)

if predict_btn and not invalid_overs:
    # Match situation terminal edge-case checks
    if current_score >= target_score:
        st.success(f"🎉 **{batting_team}** has already reached the target score of {target_score} and WON the match!")
    elif wickets_lost >= 10:
        st.error(f"💥 **{batting_team}** is All Out (10 wickets down)! **{bowling_team}** WINS by {runs_left} runs!")
    elif balls_left <= 0:
        st.error(f"⌛ 20 overs finished! **{batting_team}** fell short by {runs_left} runs. **{bowling_team}** WINS!")
    elif pipe is None:
        st.error("Model pipeline is not loaded. Please train the model first.")
    else:
        # Prepare input dataframe strictly matching pipeline feature order
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [host_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'total_runs_x': [target_score],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Generate probabilities from Logistic Regression
        result_proba = pipe.predict_proba(input_df)
        bowling_prob = round(result_proba[0][0] * 100, 1)
        batting_prob = round(result_proba[0][1] * 100, 1)

        # Dynamic Verdict Narrative
        if batting_prob >= 75:
            verdict = f"🔥 {batting_team} is in commanding control of this chase!"
        elif batting_prob >= 55:
            verdict = f"⚡ {batting_team} holds a slight competitive edge, but wickets are key."
        elif batting_prob >= 45:
            verdict = f"⚖️ Dead Heat! A thrilling 50-50 contest hanging on the next over."
        elif batting_prob >= 25:
            verdict = f"🛡️ {bowling_team} holds defensive advantage with bowling pressure."
        else:
            verdict = f"🔒 {bowling_team} is dominating! High required rate and depleted wickets."

        st.markdown(f'<div class="verdict-banner">{verdict}</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"""
            <div class="prob-card prob-card-batting">
                <div class="team-name">🏏 {batting_team}</div>
                <div class="prob-pct">{batting_prob}%</div>
                <div style="color: #93C5FD; font-size: 0.95rem; font-weight: 600;">CHANCE OF WINNING</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(batting_prob / 100.0)

        with res_col2:
            st.markdown(f"""
            <div class="prob-card prob-card-bowling">
                <div class="team-name">🎯 {bowling_team}</div>
                <div class="prob-pct">{bowling_prob}%</div>
                <div style="color: #F472B6; font-size: 0.95rem; font-weight: 600;">CHANCE OF WINNING</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(bowling_prob / 100.0)

        # Strategic Analysis Accordion
        with st.expander("🔍 Match Dynamics & Feature Breakdown", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                rate_diff = rrr - crr
                diff_text = f"+{rate_diff:.2f} rpo" if rate_diff > 0 else f"{rate_diff:.2f} rpo"
                st.metric("Rate Differential (RRR - CRR)", diff_text, delta=-round(rate_diff, 2), delta_color="inverse")
            with f2:
                balls_per_wicket = round(balls_left / max(1, wickets_left), 1)
                st.metric("Balls per Wicket in Hand", f"{balls_per_wicket} balls")
            with f3:
                rpb = round(runs_left / max(1, balls_left), 2)
                st.metric("Runs Needed Per Ball", f"{rpb} runs/ball")

            st.caption("ℹ️ *Probability computed using Scikit-Learn Logistic Regression with OneHotEncoded team histories and venue ground distributions.*")


# ==========================================
# 10. Footer & Credits
# ==========================================
st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem;">
    <strong>IPL Win Probability Predictor</strong> • NTCC Major Machine Learning Project • Built with Python & Streamlit<br>
    Features: <code>runs_left</code>, <code>balls_left</code>, <code>wickets_left</code>, <code>crr</code>, <code>rrr</code>, <code>venue_city</code>
</div>
""", unsafe_allow_html=True)

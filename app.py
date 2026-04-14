import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----- Config -----
LOOKAHEAD_SECONDS = 30
FEATURE_WINDOW = 30
OFF_THRESHOLD = 0.5
GRACE_SECONDS = 5

# Initialize session state for "Hardware" simulation
if 'motion_events' not in st.session_state:
    st.session_state.motion_events = []
if 'light_on' not in st.session_state:
    st.session_state.light_on = False

# --- UI Layout ---
st.title("💡 Smart Energy Saver MVP")
st.subheader("AI-Driven Lighting Control")

col1, col2 = st.columns(2)
with col1:
    light_placeholder = st.empty()
with col2:
    prob_placeholder = st.empty()

chart_placeholder = st.line_chart(np.zeros(20))

# --- Functions ---
def get_motion_count():
    now = time.time()
    # Filter events in the last 30s
    st.session_state.motion_events = [t for t in st.session_state.motion_events if now - t < FEATURE_WINDOW]
    return len(st.session_state.motion_events)

@st.cache_resource
def get_mock_model():
    """Returns a simple pre-trained style model for the demo"""
    X = np.array([[0, 0, 0], [1, 0.5, 0.5], [5, 1, 0], [10, -1, 0]])
    y = [0, 0, 1, 1]
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    return clf

# --- Main Logic ---
clf = get_mock_model()
history = []

st.write("### Live Logs")
log_box = st.empty()

# Run the live simulation
while True:
    now = time.time()
    
    # 1. Simulate random motion (chance of person moving)
    if np.random.random() > 0.8:
        st.session_state.motion_events.append(now)
    
    # 2. Extract Features
    m_count = get_motion_count()
    h_sin = np.sin(2 * np.pi * (now % 60) / 60)
    h_cos = np.cos(2 * np.pi * (now % 60) / 60)
    
    # 3. Predict
    prob = float(clf.predict_proba([[m_count, h_sin, h_cos]])[0][1])
    
    # 4. Decision Logic (Fixed the line 128 error)
    last_motion_age = now - st.session_state.motion_events[-1] if st.session_state.motion_events else 999
    
    if last_motion_age < GRACE_SECONDS:
        st.session_state.light_on = True
    elif prob < OFF_THRESHOLD:
        st.session_state.light_on = False
    elif prob > 0.7:
        st.session_state.light_on = True

    # 5. Update UI
    status_emoji = "🟡 ON" if st.session_state.light_on else "⚫ OFF"
    light_placeholder.metric("Light State", status_emoji)
    prob_placeholder.metric("Occupancy Prob", f"{prob*100:.1f}%")
    
    history.append(prob)
    chart_placeholder.line_chart(history[-20:]) # Show last 20 samples
    
    log_box.text(f"[{time.strftime('%H:%M:%S')}] Count: {m_count} | Prob: {prob:.2f}")
    
    time.sleep(1)

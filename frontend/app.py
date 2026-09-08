import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os
import sys
from pathlib import Path

# Streamlit Community Cloud runs only this app. If no backend URL is supplied,
# import the local forecast modules directly so this repository deploys alone.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_URL = os.getenv("BACKEND_URL", "").rstrip("/")
if not BACKEND_URL:
    from backend.predict import predict_next_24h, predict_next_72h, predict_next_168h
    from backend.fetch import get_recent_data
    from backend.training import MODEL_PATH, train as train_candidate

st.set_page_config(
    page_title="Bakkhali Weather Log",
    page_icon="⚓",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Visual identity: an antique brass instrument kept in a dark-panelled study —
# gilt hairlines, glass panels with real depth, serif engraving for numerals.
# The dark base palette itself comes from .streamlit/config.toml so that
# native widgets (sidebar, inputs, popovers) theme correctly, not just the
# custom HTML blocks below.
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500;1,600&family=Jost:wght@400;500;600&display=swap');

    :root {
        --bg: #0A0F14;
        --panel: #101820;
        --gold: #C6A15B;
        --gold-bright: #E4C583;
        --seaglass: #4B6F64;
        --oxblood: #7C2D2D;
        --ivory: #EDE7D8;
        --ivory-dim: rgba(237,231,216,0.62);
        --hairline: rgba(198,161,91,0.28);
    }

    .stApp {
        background: radial-gradient(ellipse at 50% -10%, #16222A 0%, #0A0F14 55%, #06090C 100%);
    }

    body, [class*="css"] { font-family: 'Jost', sans-serif; }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }

    hr { border: none; border-top: 1px solid var(--hairline); margin: 1.6rem 0; }

    /* ---- Corner-bracket framing, reused on the header, cards and panel --- */
    .framed { position: relative; }
    .framed::before, .framed::after {
        content: "";
        position: absolute;
        width: 13px;
        height: 13px;
        border: 1px solid var(--gold);
        opacity: 0.85;
    }
    .framed::before { top: 6px; left: 6px; border-width: 1px 0 0 1px; }
    .framed::after { bottom: 6px; right: 6px; border-width: 0 1px 1px 0; }

    /* ---- Header -------------------------------------------------------- */
    .coast-header {
        text-align: center;
        background: linear-gradient(180deg, rgba(22,34,42,0.7), rgba(10,15,20,0.85));
        backdrop-filter: blur(14px);
        border: 1px solid var(--hairline);
        border-radius: 2px;
        padding: 2.6rem 2rem 2.1rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 24px 48px rgba(0,0,0,0.45);
    }
    .coast-header h1 {
        font-style: italic;
        font-size: 2.5rem;
        margin: 0;
        color: var(--ivory);
    }
    .coast-header .rule {
        width: 64px;
        height: 1px;
        background: var(--gold);
        margin: 0.9rem auto 0;
        opacity: 0.85;
    }
    .coast-header p {
        font-size: 0.98rem;
        color: var(--ivory-dim);
        max-width: 50ch;
        margin: 0.9rem auto 0;
    }

    /* ---- Metric plaques -------------------------------------------------- */
    .gauge-card {
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 2px;
        padding: 1.3rem 1.2rem 1.1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 12px 26px rgba(0,0,0,0.35);
    }
    .gauge-card .g-label {
        font-size: 0.8rem;
        font-style: italic;
        color: var(--ivory-dim);
        margin-bottom: 0.4rem;
    }
    .gauge-card .g-value {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 500;
        font-size: 2.1rem;
        letter-spacing: 0.02em;
        color: var(--gold-bright);
    }

    /* ---- Empty-state panel (log-entry style) ----------------------------- */
    .panel {
        display: flex;
        flex-wrap: wrap;
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 2px;
    }
    .panel .panel-item {
        flex: 1 1 220px;
        padding: 1.7rem 1.5rem;
        border-right: 1px solid var(--hairline);
    }
    .panel .panel-item:last-child { border-right: none; }
    .panel .panel-item h4 {
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0 0 0.4rem;
        color: var(--ivory);
    }
    .panel .panel-item p {
        font-size: 0.86rem;
        color: var(--ivory-dim);
        line-height: 1.55;
        margin: 0;
    }

    /* ---- Buttons ---------------------------------------------------------*/
    .stButton>button {
        font-family: 'Jost', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em;
        border-radius: 2px !important;
        border: 1px solid var(--gold) !important;
        background: transparent !important;
        color: var(--gold-bright) !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover { background: var(--gold) !important; color: #0A0F14 !important; }
    .stButton>button[kind="primary"] { background: var(--gold) !important; color: #0A0F14 !important; }
    .stButton>button[kind="primary"]:hover { background: var(--gold-bright) !important; }

    /* ---- Misc chrome -------------------------------------------------------*/
    [data-testid="stSidebar"] img { border-radius: 2px; border: 1px solid var(--hairline); }
    [data-testid="stAlert"] {
        background: rgba(16,24,32,0.65) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 2px !important;
    }
    .streamlit-expanderHeader { font-family: 'Cormorant Garamond', serif; font-weight: 600; font-style: italic; }

    .coast-footer {
        text-align: center;
        padding: 1.4rem 0 0.4rem;
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        font-size: 0.92rem;
        color: var(--ivory-dim);
        border-top: 1px solid var(--hairline);
        margin-top: 2.2rem;
    }
    </style>
""", unsafe_allow_html=True)

COMPASS_SVG = """
<svg width="38" height="38" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="20" r="17" stroke="#C6A15B" stroke-width="1"/>
  <path d="M20 6 L23 20 L20 34 L17 20 Z" stroke="#C6A15B" stroke-width="1" fill="none"/>
  <path d="M6 20 L20 17 L34 20 L20 23 Z" stroke="#C6A15B" stroke-width="1" fill="none"/>
  <circle cx="20" cy="20" r="2" fill="#C6A15B"/>
</svg>
"""

st.markdown(
    f'<div class="coast-header framed">'
    f'{COMPASS_SVG}'
    f'<h1>Bakkhali Weather Log</h1>'
    f'<div class="rule"></div>'
    f'<p>Hourly conditions for Bakkhali Beach, West Bengal, kept by a locally '
    f'trained forecasting model — temperature, humidity, wind and radiation.</p>'
    f'</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Bakkhali_sea_beach_3.jpg/800px-Bakkhali_sea_beach_3.jpg",
             caption="Bakkhali Beach")

    st.header("Forecast controls")

    horizon = st.selectbox(
        "Forecast range",
        options=["24h", "72h", "168h"],
        format_func=lambda value: {"24h": "Next 24 hours", "72h": "Next 72 hours (3 days)", "168h": "Next 168 hours (7 days)"}[value],
    )
    st.info(f"Displaying the {horizon} forecast")

    if not BACKEND_URL and not MODEL_PATH.exists():
        st.warning("Train the custom model first using the button below. The old saved model is intentionally ignored.")

    st.markdown("---")

    if st.button("Get latest predictions", type="primary", use_container_width=True):
        with st.spinner(f"Fetching {horizon} predictions..."):
            try:
                if not BACKEND_URL and not MODEL_PATH.exists():
                    st.error("No custom model is trained yet. Click 'Train / retrain custom model' first.")
                    st.stop()
                if BACKEND_URL:
                    response = requests.get(f"{BACKEND_URL}/api/predict/{horizon}", timeout=30)
                    if response.status_code != 200:
                        st.error(f"Failed to connect to backend (status: {response.status_code})")
                        st.stop()
                    data = response.json()
                    if not data["success"]:
                        st.error(f"Error: {data.get('error', 'Unknown error')}")
                        st.stop()
                    rows = data.get("predictions", data.get("hourly", []))
                else:
                    history = get_recent_data(hours=360)
                    rows = {
                        "24h": lambda: predict_next_24h(history),
                        "72h": lambda: predict_next_72h(history)["hourly"],
                        "168h": lambda: predict_next_168h(history)["hourly"],
                    }[horizon]()
                st.session_state.predictions = rows
                st.session_state.horizon = horizon
                st.session_state.last_update = datetime.datetime.now()
                st.success("Predictions updated")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend at " + BACKEND_URL)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.caption("Custom ML model — no future forecast values are used")
    if BACKEND_URL:
        st.info("Candidate-model training runs on the backend host through its scheduler.")
    elif st.button("Train / retrain custom model", use_container_width=True):
        with st.spinner("Downloading historical data and training the custom next-hour model. This can take a few minutes..."):
            try:
                report = train_candidate(start_date="2022-01-01")
                st.session_state.training_report = report
                st.success("Custom model trained, evaluated, and activated.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

    if "last_update" in st.session_state:
        st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    if "training_report" in st.session_state:
        report = st.session_state.training_report
        st.caption(f"Active custom model: {report['model']} · {report['train_rows']} training rows")


if "predictions" in st.session_state and st.session_state.predictions:
    predictions = st.session_state.predictions
    selected_horizon = st.session_state.get("horizon", "24h")

    st.header(f"Next {selected_horizon} forecast")

    df = pd.DataFrame(predictions)

    col1, col2, col3, col4 = st.columns(4)
    gauges = [
        (col1, "Current temperature", f"{df['Temperature (°C)'].iloc[0]:.1f}°C"),
        (col2, "Current humidity", f"{df['Humidity (%)'].iloc[0]:.1f}%"),
        (col3, "Wind speed", f"{df['Wind Speed (m/s)'].iloc[0]:.1f} m/s"),
        (col4, "Pressure", f"{df['Pressure (kPa)'].iloc[0]:.1f} kPa"),
    ]
    for col, label, value in gauges:
        with col:
            st.markdown(
                f'<div class="gauge-card framed"><div class="g-label">{label}</div>'
                f'<div class="g-value">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Temperature', 'Humidity & precipitation', 'Wind speed & radiation'),
        vertical_spacing=0.12
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Temperature (°C)'],
                  mode='lines+markers', name='Temperature',
                  line=dict(color='#C6A15B', width=2.5),
                  marker=dict(size=6)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Humidity (%)'],
                  mode='lines+markers', name='Humidity',
                  line=dict(color='#4B6F64', width=2.5),
                  marker=dict(size=6)),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=df['datetime'], y=df['Precipitation (mm/hr)'],
              name='Precipitation', marker_color='#5C7A8A',
              opacity=0.6),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Wind Speed (m/s)'],
                  mode='lines+markers', name='Wind Speed',
                  line=dict(color='#8C5A3C', width=2.5),
                  marker=dict(size=6)),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Radiation (W/m²)'],
                  mode='lines+markers', name='Radiation',
                  line=dict(color='#D9C08C', width=2.5),
                  marker=dict(size=6)),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='#0F171C',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Jost, sans-serif", color="#EDE7D8", size=12),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=60, l=50, r=30, b=40),
    )
    fig.update_xaxes(title_text="Time", row=3, col=1, gridcolor='rgba(198,161,91,0.10)')
    fig.update_xaxes(gridcolor='rgba(198,161,91,0.10)')
    fig.update_yaxes(gridcolor='rgba(198,161,91,0.10)', zerolinecolor='rgba(198,161,91,0.16)')
    fig.update_annotations(font=dict(family="Cormorant Garamond, serif", size=15, color="#EDE7D8"))

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View hourly breakdown"):
        st.dataframe(
            df.style.format({
                'Temperature (°C)': '{:.1f}°C',
                'Humidity (%)': '{:.1f}%',
                'Wind Speed (m/s)': '{:.1f} m/s',
                'Pressure (kPa)': '{:.1f} kPa',
                'Precipitation (mm/hr)': '{:.2f} mm',
                'Cloud Coverage (%)': '{:.1f}%',
                'Radiation (W/m²)': '{:.1f} W/m²'
            }),
            use_container_width=True
        )

else:
    st.info("Choose a range and click 'Get latest predictions' in the sidebar.")

    st.markdown(
        '<div class="panel framed">'
        '<div class="panel-item"><h4>Temperature</h4>'
        '<p>Hourly forecast temperatures for the selected range.</p></div>'
        '<div class="panel-item"><h4>Humidity &amp; rain</h4>'
        '<p>Track humidity levels alongside expected precipitation.</p></div>'
        '<div class="panel-item"><h4>Wind &amp; radiation</h4>'
        '<p>Monitor wind speed and solar radiation together.</p></div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="coast-footer">Bakkhali Weather Log · data from Open-Meteo</div>',
    unsafe_allow_html=True
)
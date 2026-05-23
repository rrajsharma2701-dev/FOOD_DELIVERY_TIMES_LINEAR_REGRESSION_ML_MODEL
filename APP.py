import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeliveryIQ · ML Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS  (3-D card look, neon accents, glassmorphism)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── root palette ── */
:root {
    --bg:       #09090f;
    --surface:  #12121e;
    --card:     #1a1a2e;
    --border:   rgba(120,80,255,0.25);
    --neon:     #7c4dff;
    --neon2:    #00e5ff;
    --neon3:    #ff4081;
    --text:     #e8e8f0;
    --muted:    #7070a0;
    --success:  #00e676;
    --warn:     #ffab40;
}

/* ── base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── remove streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #12122a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f0f2d 0%, #1a0a3e 40%, #0a1a3e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.4rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 8px 32px rgba(124,77,255,0.18),
        0 2px 0 rgba(124,77,255,0.5) inset,
        0 -2px 0 rgba(0,229,255,0.15) inset;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(0,229,255,0.06), transparent);
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    color: var(--neon2);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fff 0%, var(--neon2) 60%, var(--neon) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
}

/* ── 3-D metric cards ── */
.metric-row { display: flex; gap: 1.2rem; margin-bottom: 1.8rem; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 10px 30px rgba(0,0,0,0.5),
        0 1px 0 rgba(255,255,255,0.06) inset;
    transition: transform .25s, box-shadow .25s;
}
.metric-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, var(--neon));
    border-radius: 16px 16px 0 0;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 40px rgba(0,0,0,0.6), 0 0 0 1px var(--border);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.3rem;
}
.metric-icon {
    position: absolute;
    top: 1.2rem; right: 1.4rem;
    font-size: 1.8rem;
    opacity: 0.18;
}

/* ── section headers ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.2rem;
}
.section-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--neon);
    box-shadow: 0 0 10px var(--neon);
    flex-shrink: 0;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #fff;
}

/* ── glass panel ── */
.glass {
    background: rgba(26,26,46,0.7);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 1.4rem;
}

/* ── prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #0f1f3e, #1a0a3e);
    border: 1px solid rgba(0,229,255,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow:
        0 0 60px rgba(0,229,255,0.12),
        0 0 0 1px rgba(0,229,255,0.08);
}
.pred-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: var(--neon2);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.pred-value {
    font-size: 5rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--neon2), var(--neon));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.pred-unit {
    font-size: 1.2rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* ── sliders & widgets ── */
[data-testid="stSlider"] > div { color: var(--text) !important; }
.stSlider > label { color: var(--muted) !important; font-size: 0.85rem !important; }
.stSelectbox label, .stRadio label { color: var(--muted) !important; font-size: 0.85rem !important; }

/* ── streamlit elements ── */
div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

/* ── tab style ── */
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--neon2) !important;
    border-bottom: 2px solid var(--neon2) !important;
}

/* ── horizontal rule ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── info/warning boxes ── */
.stAlert { border-radius: 12px !important; }

/* ── number badge ── */
.badge {
    display: inline-block;
    background: rgba(124,77,255,0.18);
    border: 1px solid rgba(124,77,255,0.35);
    color: var(--neon);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 30px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DATA LOADING & MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Food_Delivery_Times.csv")

    # fill nulls
    for col in ["Weather", "Traffic_Level", "Time_of_Day"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(df["Courier_Experience_yrs"].mode()[0])

    X = df.drop(["Delivery_Time_min", "Order_ID"], axis=1)
    y = df["Delivery_Time_min"]

    X_encoded = pd.get_dummies(X, columns=["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"], drop_first=True)
    X_encoded = X_encoded.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return df, model, X_encoded, X_train, X_test, y_train, y_test, y_pred, r2, mae, rmse

df, model, X_encoded, X_train, X_test, y_train, y_test, y_pred, r2, mae, rmse = load_and_train()


# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#12121e",
    "axes.facecolor":    "#12121e",
    "axes.edgecolor":    "#2a2a4a",
    "axes.labelcolor":   "#8080b0",
    "xtick.color":       "#8080b0",
    "ytick.color":       "#8080b0",
    "text.color":        "#e8e8f0",
    "grid.color":        "#1e1e38",
    "grid.linestyle":    "--",
    "font.family":       "monospace",
})

NEON   = "#7c4dff"
NEON2  = "#00e5ff"
NEON3  = "#ff4081"
GOLD   = "#ffab40"


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.2rem 0 1.5rem;'>
        <div style='font-size:2.8rem'>🚀</div>
        <div style='font-weight:800; font-size:1.1rem; color:#fff;'>DeliveryIQ</div>
        <div style='font-size:0.72rem; color:#7070a0; letter-spacing:2px;'>ML PREDICTION ENGINE</div>
    </div>
    <hr style='margin:0 0 1.2rem;'/>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "NAVIGATION",
        ["🏠  Overview", "🔮  Predict", "📊  Insights", "⚙️  Model Details"],
        label_visibility="collapsed",
    )
    page = nav.split("  ")[1]

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='glass' style='padding:1rem;'>
        <div class='metric-label'>MODEL STATUS</div>
        <div style='color:#00e676; font-weight:700; font-size:0.95rem;'>● LIVE</div>
        <div class='metric-label' style='margin-top:.8rem;'>ALGORITHM</div>
        <div style='color:#fff; font-size:0.85rem;'>Linear Regression</div>
        <div class='metric-label' style='margin-top:.8rem;'>TRAINING ROWS</div>
        <div style='color:#fff; font-size:0.85rem;'>{len(X_train):,}</div>
        <div class='metric-label' style='margin-top:.8rem;'>FEATURES</div>
        <div style='color:#fff; font-size:0.85rem;'>{X_encoded.shape[1]}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-tag'>// food delivery · linear regression · v1.0</div>
    <h1>DeliveryIQ<br/>Prediction Engine</h1>
    <p>Real-time delivery time forecasting powered by machine learning &nbsp;|&nbsp; 1,000 orders trained</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  METRIC CARDS (always visible)
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='metric-row'>
    <div class='metric-card' style='--accent:{NEON2};'>
        <div class='metric-icon'>📈</div>
        <div class='metric-label'>R² Score</div>
        <div class='metric-value'>{r2:.4f}</div>
        <div class='metric-sub'>Model accuracy</div>
    </div>
    <div class='metric-card' style='--accent:{GOLD};'>
        <div class='metric-icon'>⏱️</div>
        <div class='metric-label'>Mean Abs Error</div>
        <div class='metric-value'>{mae:.2f}</div>
        <div class='metric-sub'>minutes off on average</div>
    </div>
    <div class='metric-card' style='--accent:{NEON3};'>
        <div class='metric-icon'>📉</div>
        <div class='metric-label'>RMSE</div>
        <div class='metric-value'>{rmse:.2f}</div>
        <div class='metric-sub'>root mean sq error</div>
    </div>
    <div class='metric-card' style='--accent:{NEON};'>
        <div class='metric-icon'>🗄️</div>
        <div class='metric-label'>Dataset Size</div>
        <div class='metric-value'>{len(df):,}</div>
        <div class='metric-sub'>total orders</div>
    </div>
    <div class='metric-card' style='--accent:#00e676;'>
        <div class='metric-icon'>🚗</div>
        <div class='metric-label'>Avg Delivery</div>
        <div class='metric-value'>{df["Delivery_Time_min"].mean():.0f}<span style='font-size:1rem'>m</span></div>
        <div class='metric-sub'>baseline average</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "Overview":

    st.markdown("""<div class='section-head'><div class='section-dot'></div><div class='section-title'>Dataset Preview</div></div>""", unsafe_allow_html=True)

    st.dataframe(
        df.head(20),
        use_container_width=True,
        height=320,
    )

    col1, col2 = st.columns(2)

    # Distribution
    with col1:
        st.markdown("""<div class='section-head'><div class='section-dot' style='background:#00e5ff;box-shadow:0 0 10px #00e5ff;'></div><div class='section-title'>Delivery Time Distribution</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        n, bins, patches = ax.hist(df["Delivery_Time_min"], bins=35, color=NEON, alpha=0.75, edgecolor="#09090f", linewidth=0.5)
        for patch in patches:
            patch.set_facecolor(NEON)
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(df["Delivery_Time_min"].min(), df["Delivery_Time_min"].max(), 300)
        kde = gaussian_kde(df["Delivery_Time_min"].dropna())
        ax2 = ax.twinx()
        ax2.plot(kde_x, kde(kde_x), color=NEON2, lw=2.5, label="KDE")
        ax2.set_yticks([])
        ax2.spines[:].set_visible(False)
        ax.set_xlabel("Delivery Time (min)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Delivery Time", fontsize=11, fontweight="bold", color="#fff")
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Traffic box
    with col2:
        st.markdown("""<div class='section-head'><div class='section-dot' style='background:#ff4081;box-shadow:0 0 10px #ff4081;'></div><div class='section-title'>Delivery by Traffic Level</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        order = ["Low", "Medium", "High"]
        data_by_traffic = [df[df["Traffic_Level"] == t]["Delivery_Time_min"].dropna() for t in order]
        bp = ax.boxplot(data_by_traffic, labels=order, patch_artist=True,
                        medianprops=dict(color="#fff", linewidth=2.5),
                        whiskerprops=dict(color=NEON, linewidth=1.5),
                        capprops=dict(color=NEON, linewidth=1.5),
                        flierprops=dict(markerfacecolor=NEON3, marker="o", markersize=3, alpha=0.5))
        colors = [NEON2, GOLD, NEON3]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_ylabel("Delivery Time (min)")
        ax.set_title("Time vs Traffic Level", fontsize=11, fontweight="bold", color="#fff")
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""<div class='section-head'><div class='section-dot' style='background:#ffab40;box-shadow:0 0 10px #ffab40;'></div><div class='section-title'>Distance vs Delivery Time</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(df["Distance_km"], df["Delivery_Time_min"],
                        c=df["Delivery_Time_min"], cmap="plasma",
                        s=12, alpha=0.5, linewidths=0)
        m, b = np.polyfit(df["Distance_km"].dropna(), df["Delivery_Time_min"].dropna(), 1)
        x_line = np.linspace(df["Distance_km"].min(), df["Distance_km"].max(), 100)
        ax.plot(x_line, m * x_line + b, color=NEON2, lw=2.5, label=f"Trend (slope={m:.2f})")
        ax.legend(fontsize=8)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Delivery Time (min)")
        ax.set_title("Distance vs Delivery Time", fontsize=11, fontweight="bold", color="#fff")
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Delivery Time")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown("""<div class='section-head'><div class='section-dot' style='background:#7c4dff;box-shadow:0 0 10px #7c4dff;'></div><div class='section-title'>Vehicle Type Breakdown</div></div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = df["Vehicle_Type"].value_counts()
        wedge_colors = [NEON, NEON2, NEON3]
        wedges, texts, autotexts = ax.pie(
            vc.values,
            labels=vc.index,
            autopct="%1.1f%%",
            colors=wedge_colors,
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(edgecolor="#09090f", linewidth=2.5, width=0.6),
        )
        for at in autotexts:
            at.set_color("#fff")
            at.set_fontsize(9)
        for t in texts:
            t.set_color("#c0c0e0")
        ax.set_title("Vehicle Type Distribution", fontsize=11, fontweight="bold", color="#fff")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────────────────────
#  PAGE: PREDICT
# ─────────────────────────────────────────────────────────────
elif page == "Predict":
    st.markdown("""<div class='section-head'><div class='section-dot' style='background:#00e5ff;box-shadow:0 0 10px #00e5ff;'></div><div class='section-title'>Live Delivery Time Predictor</div></div>""", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        distance   = st.slider("🗺️  Distance (km)",              0.5, 20.0, 8.0, 0.1)
        prep_time  = st.slider("🍳  Preparation Time (min)",      1,   60,   15,   1)
        experience = st.slider("👤  Courier Experience (years)",  0,   10,   3,    1)

        c1, c2 = st.columns(2)
        with c1:
            weather = st.selectbox("🌤️  Weather",       ["Clear","Windy","Foggy","Rainy","Snowy"])
            time_of_day = st.selectbox("🕐  Time of Day",  ["Morning","Afternoon","Evening","Night"])
        with c2:
            traffic = st.selectbox("🚦  Traffic Level", ["Low","Medium","High"])
            vehicle = st.selectbox("🛵  Vehicle Type",  ["Scooter","Bike","Car"])

        predict_btn = st.button("⚡  PREDICT DELIVERY TIME", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if predict_btn:
            # Build feature row exactly matching training schema
            input_dict = {
                "Distance_km": distance,
                "Preparation_Time_min": prep_time,
                "Courier_Experience_yrs": experience,
                "Weather": weather,
                "Traffic_Level": traffic,
                "Time_of_Day": time_of_day,
                "Vehicle_Type": vehicle,
            }
            input_df = pd.DataFrame([input_dict])
            input_encoded = pd.get_dummies(input_df, columns=["Weather","Traffic_Level","Time_of_Day","Vehicle_Type"], drop_first=True)
            # align columns
            input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0).astype(int)
            prediction = model.predict(input_encoded)[0]
            prediction = max(5, prediction)  # clamp

            # confidence band ±MAE
            low  = max(0, prediction - mae)
            high = prediction + mae

            st.markdown(f"""
            <div class='pred-box'>
                <div class='pred-label'>// estimated delivery time</div>
                <div class='pred-value'>{prediction:.1f}</div>
                <div class='pred-unit'>minutes</div>
                <div style='margin-top:1.2rem; font-size:0.8rem; color:#5050a0;'>
                    confidence range &nbsp;·&nbsp;
                    <span style='color:#ffab40'>{low:.0f} – {high:.0f} min</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # mini gauge
            st.markdown("<br/>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 2.2))
            max_time = 160
            norm_val = min(prediction / max_time, 1.0)
            bar_color = NEON2 if norm_val < 0.4 else GOLD if norm_val < 0.7 else NEON3
            ax.barh(["ETA"], [max_time], color="#1e1e38", height=0.45)
            ax.barh(["ETA"], [prediction], color=bar_color, height=0.45,
                    left=0)
            ax.axvline(x=prediction, color="#fff", linewidth=2, linestyle="--", alpha=0.7)
            ax.text(prediction + 2, 0, f"{prediction:.0f} min", va="center", color="#fff", fontsize=11, fontweight="bold")
            ax.set_xlim(0, max_time)
            ax.set_xlabel("Delivery Time (min)")
            ax.spines[:].set_visible(False)
            ax.set_title("ETA Gauge", color="#fff", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # breakdown cards
            st.markdown(f"""
            <div style='display:flex;gap:.8rem;margin-top:.8rem;'>
                <div class='glass' style='flex:1;text-align:center;padding:.8rem;'>
                    <div class='metric-label'>Distance</div>
                    <div style='font-size:1.3rem;font-weight:800;color:#00e5ff;'>{distance} km</div>
                </div>
                <div class='glass' style='flex:1;text-align:center;padding:.8rem;'>
                    <div class='metric-label'>Prep Time</div>
                    <div style='font-size:1.3rem;font-weight:800;color:#ffab40;'>{prep_time} min</div>
                </div>
                <div class='glass' style='flex:1;text-align:center;padding:.8rem;'>
                    <div class='metric-label'>Traffic</div>
                    <div style='font-size:1.3rem;font-weight:800;color:#ff4081;'>{traffic}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='glass' style='text-align:center; padding:3rem 2rem;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>⚡</div>
                <div style='font-size:1.1rem; font-weight:700; color:#7070a0;'>
                    Configure the parameters on the left<br/>and hit <span style='color:#00e5ff;'>PREDICT</span> to get your ETA
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PAGE: INSIGHTS
# ─────────────────────────────────────────────────────────────
elif page == "Insights":
    st.markdown("""<div class='section-head'><div class='section-dot' style='background:#ffab40;box-shadow:0 0 10px #ffab40;'></div><div class='section-title'>Actual vs Predicted</div></div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.45, s=18,
                   c=np.abs(y_test.values - y_pred),
                   cmap="plasma", linewidths=0)
        lims = [min(y_test.min(), y_pred.min()) - 5,
                max(y_test.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, color=NEON2, lw=2, linestyle="--", label="Perfect fit")
        ax.set_xlabel("Actual (min)")
        ax.set_ylabel("Predicted (min)")
        ax.set_title("Actual vs Predicted Delivery Time", fontsize=11, fontweight="bold", color="#fff")
        ax.legend(fontsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        residuals = y_test.values - y_pred
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(residuals, bins=40, color=NEON, alpha=0.75, edgecolor="#09090f")
        ax.axvline(0, color=NEON2, linewidth=2, linestyle="--", label="Zero error")
        ax.set_xlabel("Residual (min)")
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution", fontsize=11, fontweight="bold", color="#fff")
        ax.legend(fontsize=8)
        ax.spines[:].set_visible(False)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Feature importances (coefficients)
    st.markdown("""<div class='section-head'><div class='section-dot' style='background:#7c4dff;box-shadow:0 0 10px #7c4dff;'></div><div class='section-title'>Feature Importance (Coefficients)</div></div>""", unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(coef_df) * 0.36)))
    colors = [NEON3 if c < 0 else NEON for c in coef_df["Coefficient"]]
    bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, alpha=0.8, height=0.65)
    ax.axvline(0, color="#5050a0", linewidth=1.5)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Linear Regression Coefficients", fontsize=12, fontweight="bold", color="#fff")
    ax.spines[:].set_visible(False)
    ax.grid(True, alpha=0.2, axis="x")
    neg_p = mpatches.Patch(color=NEON3, label="Negative (reduces time)")
    pos_p = mpatches.Patch(color=NEON, label="Positive (increases time)")
    ax.legend(handles=[pos_p, neg_p], fontsize=8, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Weather heatmap
    st.markdown("""<div class='section-head'><div class='section-dot' style='background:#00e5ff;box-shadow:0 0 10px #00e5ff;'></div><div class='section-title'>Avg Delivery Time Heatmap</div></div>""", unsafe_allow_html=True)

    pivot = df.groupby(["Weather", "Traffic_Level"])["Delivery_Time_min"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="magma",
                linewidths=0.5, linecolor="#09090f",
                ax=ax, cbar_kws={"label": "Avg Min"})
    ax.set_title("Avg Delivery Time by Weather × Traffic", fontsize=11, fontweight="bold", color="#fff")
    ax.set_xlabel("Traffic Level")
    ax.set_ylabel("Weather")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────────────────────
#  PAGE: MODEL DETAILS
# ─────────────────────────────────────────────────────────────
elif page == "Model Details":
    st.markdown("""<div class='section-head'><div class='section-dot'></div><div class='section-title'>Model Architecture</div></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='glass'>
            <div class='metric-label'>Algorithm</div>
            <div style='font-size:1.05rem;font-weight:700;color:#fff;'>Ordinary Least Squares</div>
            <div class='metric-label' style='margin-top:.8rem;'>Library</div>
            <div style='color:#fff;'>scikit-learn 1.x</div>
            <div class='metric-label' style='margin-top:.8rem;'>Intercept</div>
            <div style='color:#00e5ff;font-family:JetBrains Mono,monospace;'>{model.intercept_:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='glass'>
            <div class='metric-label'>Train / Test Split</div>
            <div style='font-size:1.05rem;font-weight:700;color:#fff;'>80% / 20%</div>
            <div class='metric-label' style='margin-top:.8rem;'>Train Samples</div>
            <div style='color:#fff;'>{len(X_train):,}</div>
            <div class='metric-label' style='margin-top:.8rem;'>Test Samples</div>
            <div style='color:#ffab40;'>{len(X_test):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='glass'>
            <div class='metric-label'>Encoding</div>
            <div style='font-size:1.05rem;font-weight:700;color:#fff;'>One-Hot (drop_first)</div>
            <div class='metric-label' style='margin-top:.8rem;'>Raw Features</div>
            <div style='color:#fff;'>7</div>
            <div class='metric-label' style='margin-top:.8rem;'>Encoded Features</div>
            <div style='color:#7c4dff;'>{X_encoded.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div class='section-head'><div class='section-dot' style='background:#ff4081;box-shadow:0 0 10px #ff4081;'></div><div class='section-title'>All Feature Coefficients</div></div>""", unsafe_allow_html=True)

    coef_display = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Coefficient": model.coef_.round(4),
        "Impact": ["↑ increases time" if c > 0 else "↓ reduces time" for c in model.coef_],
    }).sort_values("Coefficient", ascending=False).reset_index(drop=True)

    st.dataframe(coef_display, use_container_width=True, height=420)

    st.markdown("""<div class='section-head'><div class='section-dot' style='background:#00e676;box-shadow:0 0 10px #00e676;'></div><div class='section-title'>Performance Summary</div></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='glass'>
        <table style='width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:0.9rem;'>
            <tr style='border-bottom:1px solid #2a2a4a;'>
                <th style='padding:.6rem 1rem;color:#7070a0;text-align:left;'>Metric</th>
                <th style='padding:.6rem 1rem;color:#7070a0;text-align:left;'>Value</th>
                <th style='padding:.6rem 1rem;color:#7070a0;text-align:left;'>Interpretation</th>
            </tr>
            <tr style='border-bottom:1px solid #1e1e38;'>
                <td style='padding:.55rem 1rem;color:#00e5ff;'>R² Score</td>
                <td style='padding:.55rem 1rem;color:#fff;'>{r2:.6f}</td>
                <td style='padding:.55rem 1rem;color:#7070a0;'>Explains {r2*100:.1f}% of variance</td>
            </tr>
            <tr style='border-bottom:1px solid #1e1e38;'>
                <td style='padding:.55rem 1rem;color:#ffab40;'>MAE</td>
                <td style='padding:.55rem 1rem;color:#fff;'>{mae:.4f} min</td>
                <td style='padding:.55rem 1rem;color:#7070a0;'>Average prediction error</td>
            </tr>
            <tr>
                <td style='padding:.55rem 1rem;color:#ff4081;'>RMSE</td>
                <td style='padding:.55rem 1rem;color:#fff;'>{rmse:.4f} min</td>
                <td style='padding:.55rem 1rem;color:#7070a0;'>Penalises large errors more</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ── footer ──
st.markdown("""
<div style='text-align:center;margin-top:3rem;padding:1rem;color:#3030508;font-size:0.75rem;letter-spacing:2px;'>
    DELIVERYIQ &nbsp;·&nbsp; LINEAR REGRESSION ML ENGINE &nbsp;·&nbsp; BUILT WITH STREAMLIT
</div>
""", unsafe_allow_html=True)
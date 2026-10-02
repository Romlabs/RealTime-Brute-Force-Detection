# -*- coding: utf-8 -*-
"""
Real-Time Brute-Force Detection via Predictive Machine Learning
Streamlit Cloud-safe + Visually Enhanced UI
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, classification_report
)
from scipy.stats import randint

warnings.filterwarnings("ignore")

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Real-Time Brute-Force Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# CUSTOM CSS — the visual magic
# ==================================================================
st.markdown("""
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ---------- Hero banner ---------- */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border-radius: 18px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -50%; right: -10%;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(56,189,248,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* ---------- KPI metric cards ---------- */
    .kpi-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.25s ease;
        height: 100%;
    }
    .kpi-card:hover {
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.10);
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }
    .kpi-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-value.accent { color: #0ea5e9; }
    .kpi-value.good   { color: #10b981; }
    .kpi-value.warn   { color: #f59e0b; }
    .kpi-value.danger { color: #ef4444; }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        letter-spacing: -0.01em;
    }
    .section-header .emoji {
        margin-right: 0.4rem;
    }

    /* ---------- Info / feature card ---------- */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        height: 100%;
        transition: all 0.2s ease;
    }
    .feature-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.12);
    }
    .feature-card h4 {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
        margin: 0.6rem 0 0.4rem 0;
    }
    .feature-card p {
        color: #64748b;
        font-size: 0.875rem;
        margin: 0;
        line-height: 1.5;
    }
    .feature-icon {
        font-size: 1.8rem;
        display: inline-block;
    }

    /* ---------- Status pill ---------- */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .pill-blue   { background:#e0f2fe; color:#0369a1; }
    .pill-green  { background:#dcfce7; color:#166534; }
    .pill-amber  { background:#fef3c7; color:#92400e; }
    .pill-red    { background:#fee2e2; color:#991b1b; }

    /* ---------- Step list ---------- */
    .step-list {
        list-style: none;
        padding: 0;
        counter-reset: step;
    }
    .step-list li {
        counter-increment: step;
        padding: 0.75rem 1rem 0.75rem 3rem;
        margin-bottom: 0.5rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        position: relative;
        font-size: 0.92rem;
        color: #334155;
        transition: all 0.2s ease;
    }
    .step-list li::before {
        content: counter(step);
        position: absolute;
        left: 0.9rem;
        top: 50%;
        transform: translateY(-50%);
        width: 26px; height: 26px;
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .step-list li:hover {
        border-color: #38bdf8;
        background: #f0f9ff;
        transform: translateX(4px);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        transition: all 0.15s ease;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(56, 189, 248, 0.15);
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.2);
    }
    section[data-testid="stSidebar"] .caption,
    section[data-testid="stSidebar"] small {
        color: #94a3b8 !important;
        font-size: 0.75rem;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(14, 165, 233, 0.4);
        color: white;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3) !important;
    }

    /* ---------- DataFrames ---------- */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* ---------- Alerts ---------- */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
    }

    /* ---------- Divider ---------- */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================
# SESSION STATE
# ==================================================================
def init_state():
    defaults = {
        "df_raw": None, "df_clean": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "X_res": None, "y_res": None,
        "feature_names": None, "scaler": None,
        "model": None, "model_name": None,
        "metrics": None, "grid_results": None, "rand_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#334155",
    "axes.titlecolor": "#0f172a",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
})
ACCENT = "#0ea5e9"
ACCENT2 = "#f59e0b"


# ==================================================================
# SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;'>
        <span style='font-size:1.6rem;'>🛡️</span>
        <span style='font-size:1.25rem; font-weight:800; letter-spacing:-0.02em;'>
            Brute-Force Detection
        </span>
    </div>
    <div style='color:#94a3b8; font-size:0.78rem; margin-bottom:1.5rem;'>
        Random Forest · SMOTE · Hyperparameter Search
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🧭 Navigation")
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📥 Load Data",
            "🧹 Data Cleaning",
            "⚖️ Class Imbalance (SMOTE)",
            "🌲 Train Model",
            "🔧 Grid Search",
            "🎲 Randomized Search",
            "📊 Feature Importance",
            "🔮 Predict",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#64748b; line-height:1.6;'>
        <b style='color:#94a3b8;'>⚙️ Environment</b><br>
        Python 3.11 · Streamlit Cloud<br>
        n_jobs=1 · cv=2 (cloud-safe)
    </div>
    """, unsafe_allow_html=True)


# ==================================================================
# REUSABLE UI HELPERS
# ==================================================================
def hero(title, subtitle, badge=None):
    badge_html = f"<div class='badge'>{badge}</div>" if badge else ""
    st.markdown(f"""
    <div class="hero">
        {badge_html}
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section(title, emoji="📌"):
    st.markdown(
        f"<div class='section-header'><span class='emoji'>{emoji}</span>{title}</div>",
        unsafe_allow_html=True,
    )


def kpi(label, value, variant=""):
    cls = f"kpi-value {variant}".strip()
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{cls}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_row(m):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Accuracy",  f"{m['Accuracy']:.2f}%",  "accent")
    with c2: kpi("Precision", f"{m['Precision']:.2f}%", "good")
    with c3: kpi("Recall",    f"{m['Recall']:.2f}%",    "good")
    with c4: kpi("F1 Score",  f"{m['F1 Score']:.2f}%",  "accent")
    with c5: kpi("PR-AUC",    f"{m['PR-AUC']:.2f}%",    "warn")


def feature_card(icon, title, body):
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <h4>{title}</h4>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================================
# ML HELPERS (unchanged logic)
# ==================================================================
@st.cache_data(show_spinner=False)
def load_from_kaggle():
    import kagglehub
    path = kagglehub.dataset_download(
        "dnkumars/cybersecurity-intrusion-detection-dataset"
    )
    csv_file = None
    for f in os.listdir(path):
        if f.endswith(".csv"):
            csv_file = os.path.join(path, f)
            break
    if csv_file is None:
        raise FileNotFoundError("No CSV found.")
    return pd.read_csv(csv_file)


def clean_dataframe(df: pd.DataFrame):
    df = df.copy()
    if "session_id" in df.columns:
        df = df.drop("session_id", axis=1)
    le = LabelEncoder()
    for col in ["protocol_type", "browser_type", "encryption_used"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df.drop_duplicates().reset_index(drop=True)


def split_and_scale(df):
    X = df.drop("attack_detected", axis=1)
    y = df["attack_detected"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    feature_names = X_train.columns.tolist()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, feature_names, scaler


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, "predict_proba") else y_pred)
    return {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "F1 Score": f1_score(y_test, y_pred, zero_division=0) * 100,
        "PR-AUC": average_precision_score(y_test, y_proba) * 100,
    }, y_pred, y_proba


def plot_confusion(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test),
                cbar=False, ax=ax,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(title, pad=12)
    plt.tight_layout()
    return fig


# ==================================================================
# PAGES
# ==================================================================

# ---------------- OVERVIEW ----------------
if page == "🏠 Overview":
    hero(
        "🛡️ Real-Time Brute-Force Detection",
        "A predictive machine-learning pipeline for detecting brute-force attacks in network sessions.",
        badge="ML Pipeline · v1.0",
    )

    section("Pipeline Overview", "🔬")
    st.markdown("""
    <ol class="step-list">
        <li><b>Load</b> the Cybersecurity Intrusion Detection dataset</li>
        <li><b>Clean</b> — label encoding, deduplication</li>
        <li><b>Balance</b> classes with SMOTE</li>
        <li><b>Train</b> a Random Forest classifier</li>
        <li><b>Tune</b> hyperparameters via Grid &amp; Randomized Search</li>
        <li><b>Visualize</b> metrics, confusion matrix &amp; feature importances</li>
    </ol>
    """, unsafe_allow_html=True)

    section("What's Under the Hood", "⚙️")
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("🌲", "Random Forest",
                     "Ensemble of decision trees — robust, interpretable, "
                     "handles mixed feature types without heavy tuning.")
    with c2:
        feature_card("⚖️", "SMOTE Balancing",
                     "Synthetic Minority Over-sampling synthesizes attack "
                     "samples so the classifier doesn't ignore rare classes.")
    with c3:
        feature_card("🔧", "Hyperparameter Search",
                     "Grid + Randomized search over tree depth, feature "
                     "subsampling, and split criteria.")

    st.info(
        "⚙️ **Cloud-safe configuration:** searches use small grids, "
        "`n_jobs=1`, and `cv=2` to fit the ~1 GB / 1 CPU Streamlit container."
    )

    section("At a Glance", "📊")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Model",     "Random Forest", "accent")
    with c2: kpi("Imbalance", "SMOTE",         "good")
    with c3: kpi("Tuning",    "Grid + Rand.",  "accent")
    with c4: kpi("Target",    "attack_detected", "warn")


# ---------------- LOAD DATA ----------------
elif page == "📥 Load Data":
    hero("📥 Load Dataset", "Fetch from Kaggle or upload your own CSV.", badge="Step 1 / 6")

    source = st.radio(
        "Choose a data source:",
        ["Kaggle (auto-download)", "Upload CSV"],
        horizontal=True,
    )

    if source == "Kaggle (auto-download)":
        if st.button("⬇️ Download from Kaggle"):
            with st.spinner("Downloading dataset..."):
                try:
                    df = load_from_kaggle()
                    st.session_state.df_raw = df
                    st.success(f"✅ Loaded **{len(df):,}** rows.")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            st.session_state.df_raw = pd.read_csv(up)
            st.success(f"✅ Loaded **{len(st.session_state.df_raw):,}** rows.")

    df = st.session_state.df_raw
    if df is not None:
        section("Dataset Summary", "📊")
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Rows", f"{df.shape[0]:,}", "accent")
        with c2: kpi("Columns", f"{df.shape[1]}", "accent")
        with c3: kpi("Duplicates", f"{int(df.duplicated().sum())}", "warn")

        section("Preview", "👀")
        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("🔎 Data types / missing values / describe"):
            t1, t2, t3 = st.tabs(["Data Types", "Missing", "Statistics"])
            with t1:
                st.dataframe(df.dtypes.astype(str).rename("dtype"),
                             use_container_width=True)
            with t2:
                st.dataframe(df.isnull().sum().rename("missing"),
                             use_container_width=True)
            with t3:
                st.dataframe(df.describe(), use_container_width=True)


# ---------------- DATA CLEANING ----------------
elif page == "🧹 Data Cleaning":
    hero("🧹 Data Cleaning",
         "Drop identifiers, label-encode categoricals, remove duplicates.",
         badge="Step 2 / 6")

    if st.session_state.df_raw is None:
        st.warning("⚠️ Load a dataset first."); st.stop()

    if st.button("▶️ Run Cleaning"):
        with st.spinner("Cleaning..."):
            st.session_state.df_clean = clean_dataframe(st.session_state.df_raw)
        st.success(f"✅ Cleaned: {len(st.session_state.df_raw):,} → "
                   f"{len(st.session_state.df_clean):,} rows")

    df = st.session_state.df_clean
    if df is not None:
        section("Cleaned Preview", "👀")
        st.dataframe(df.head(10), use_container_width=True)

        if "attack_detected" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                section("Target Distribution", "🎯")
                fig, ax = plt.subplots(figsize=(6, 4))
                vc = df["attack_detected"].value_counts()
                ax.bar(vc.index.astype(str), vc.values,
                       color=[ACCENT, ACCENT2], edgecolor="white", linewidth=1.5)
                ax.set_xlabel("attack_detected")
                ax.set_ylabel("Count")
                for i, v in enumerate(vc.values):
                    ax.text(i, v, f"{v:,}", ha="center", va="bottom",
                            fontweight="bold")
                st.pyplot(fig)
            with col2:
                section("Correlation Matrix", "🔥")
                num = df.select_dtypes(include=[np.number])
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(num.corr(), annot=True, cmap="coolwarm",
                            fmt=".2f", ax=ax, cbar_kws={"shrink": 0.8},
                            annot_kws={"fontsize": 8})
                st.pyplot(fig)

        if "session_duration" in df.columns:
            section("Outlier Check — session_duration", "📦")
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.boxplot(df["session_duration"].dropna(), vert=False,
                       patch_artist=True,
                       boxprops=dict(facecolor="#bfdbfe", edgecolor=ACCENT),
                       medianprops=dict(color="#dc2626", linewidth=2))
            ax.set_xlabel("session_duration")
            st.pyplot(fig)


# ---------------- SMOTE ----------------
elif page == "⚖️ Class Imbalance (SMOTE)":
    hero("⚖️ Class Imbalance & SMOTE",
         "Balance attack vs. benign classes with synthetic minority oversampling.",
         badge="Step 3 / 6")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Run cleaning first."); st.stop()

    if st.button("▶️ Split & Apply SMOTE"):
        with st.spinner("Splitting + SMOTE..."):
            (X_train, X_test, y_train, y_test,
             names, scaler) = split_and_scale(st.session_state.df_clean)
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.X_res = X_res
            st.session_state.y_res = y_res
            st.session_state.feature_names = names
            st.session_state.scaler = scaler
        st.success("✅ SMOTE applied.")

    if st.session_state.y_res is not None:
        section("Balancing Effect", "📊")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before SMOTE**")
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = st.session_state.y_train.value_counts()
            ax.bar(vc.index.astype(str), vc.values,
                   color=["#94a3b8", "#cbd5e1"], edgecolor="white", linewidth=1.5)
            for i, v in enumerate(vc.values):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom",
                        fontweight="bold")
            ax.set_xlabel("attack_detected"); ax.set_ylabel("Count")
            st.pyplot(fig)
        with col2:
            st.markdown("**After SMOTE**")
            fig, ax = plt.subplots(figsize=(6, 4))
            vc = st.session_state.y_res.value_counts()
            ax.bar(vc.index.astype(str), vc.values,
                   color=[ACCENT, ACCENT2], edgecolor="white", linewidth=1.5)
            for i, v in enumerate(vc.values):
                ax.text(i, v, f"{v:,}", ha="center", va="bottom",
                        fontweight="bold")
            ax.set_xlabel("attack_detected"); ax.set_ylabel("Count")
            st.pyplot(fig)

        section("Split Overview", "🧮")
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Train rows (pre-SMOTE)", f"{len(st.session_state.y_train):,}", "accent")
        with c2: kpi("Train rows (post-SMOTE)", f"{len(st.session_state.y_res):,}", "good")
        with c3: kpi("Test rows", f"{len(st.session_state.y_test):,}", "accent")
        with c4: kpi("Features", f"{len(st.session_state.feature_names)}", "warn")


# ---------------- TRAIN MODEL ----------------
elif page == "🌲 Train Model":
    hero("🌲 Train Random Forest",
         "Configure tree count and depth, then train on the SMOTE-balanced set.",
         badge="Step 4 / 6")

    if st.session_state.X_res is None:
        st.warning("⚠️ Run SMOTE first."); st.stop()

    with st.container():
        section("Hyperparameters", "🎛️")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 300, 100, 50)
        with col2:
            max_depth = st.select_slider(
                "max_depth", options=[5, 10, 20, 30, None], value=None)
        with col3:
            balanced = st.checkbox("class_weight='balanced'", value=False)

    if st.button("▶️ Train Model"):
        with st.spinner("Training Random Forest..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=42, n_jobs=1,
                class_weight="balanced" if balanced else None,
            )
            model.fit(st.session_state.X_res, st.session_state.y_res)
            metrics, y_pred, _ = evaluate_model(
                model, st.session_state.X_test, st.session_state.y_test)
            st.session_state.model = model
            st.session_state.metrics = metrics
            st.session_state.model_name = "Random Forest"
        st.success("✅ Model trained.")

    if st.session_state.metrics:
        section("Test-Set Metrics", "📈")
        metric_row(st.session_state.metrics)

        section("Confusion Matrix", "🔲")
        col1, col2 = st.columns([1, 1])
        with col1:
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                     "Random Forest"))
        with col2:
            section("Classification Report", "📋")
            st.dataframe(pd.DataFrame(classification_report(
                st.session_state.y_test, y_pred, digits=4, output_dict=True
            )).transpose(), use_container_width=True)


# ---------------- GRID SEARCH ----------------
elif page == "🔧 Grid Search":
    hero("🔧 Grid Search",
         "Exhaustive search over a compact hyperparameter grid (cloud-safe).",
         badge="Step 5a / 6")

    if st.session_state.X_res is None:
        st.warning("⚠️ Run SMOTE first."); st.stop()

    st.info("⚙️ Reduced grid · `cv=2` · `n_jobs=1` — tuned for the "
            "Streamlit Cloud container.")

    if st.button("▶️ Run Grid Search"):
        param_grid = {
            "n_estimators":      [50, 100],
            "max_features":      ["sqrt"],
            "max_depth":         [10, 20],
            "min_samples_split": [2, 5],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=1)
        gs = GridSearchCV(rf, param_grid, cv=2, n_jobs=1,
                          scoring="accuracy", verbose=0)
        with st.spinner("Running Grid Search..."):
            gs.fit(st.session_state.X_res, st.session_state.y_res)
        st.session_state.grid_results = gs

        best = gs.best_estimator_
        metrics, y_pred, _ = evaluate_model(
            best, st.session_state.X_test, st.session_state.y_test)
        st.session_state.model = best
        st.session_state.metrics = metrics
        st.session_state.model_name = "RF (Grid)"
        st.success("✅ Grid Search complete.")

    gs = st.session_state.grid_results
    if gs is not None:
        section("Best Configuration", "🏆")
        c1, c2 = st.columns([1, 2])
        with c1:
            kpi("Best CV Accuracy", f"{gs.best_score_*100:.2f}%", "good")
        with c2:
            st.markdown("**Best parameters**")
            st.json(gs.best_params_)

        section("Test-Set Metrics", "📈")
        metric_row(st.session_state.metrics)

        section("Confusion Matrix", "🔲")
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                 "Confusion Matrix — Grid Search Best"))


# ---------------- RANDOMIZED SEARCH ----------------
elif page == "🎲 Randomized Search":
    hero("🎲 Randomized Search",
         "Sample hyperparameter combinations from distributions (cloud-safe).",
         badge="Step 5b / 6")

    if st.session_state.X_res is None:
        st.warning("⚠️ Run SMOTE first."); st.stop()

    st.info("⚙️ `n_iter` capped at 20 · `cv=2` · `n_jobs=1`.")

    n_iter = st.slider("n_iter", 5, 20, 10, 5)

    if st.button("▶️ Run Randomized Search"):
        param_dist = {
            "n_estimators":      [100],
            "max_features":      ["sqrt", "log2"],
            "max_depth":         [10, 20, None],
            "min_samples_split": randint(2, 10),
            "min_samples_leaf":  randint(1, 5),
            "bootstrap":         [True, False],
            "class_weight":      [None, "balanced"],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=1)
        rs = RandomizedSearchCV(
            rf, param_dist, n_iter=n_iter, cv=2,
            scoring="f1_weighted", n_jobs=1,
            random_state=42, verbose=0, refit=True,
        )
        with st.spinner("Running Randomized Search..."):
            rs.fit(st.session_state.X_res, st.session_state.y_res)
        st.session_state.rand_results = rs

        best = rs.best_estimator_
        metrics, y_pred, _ = evaluate_model(
            best, st.session_state.X_test, st.session_state.y_test)
        st.session_state.model = best
        st.session_state.metrics = metrics
        st.session_state.model_name = "RF (Randomized)"
        st.success("✅ Randomized Search complete.")

    rs = st.session_state.rand_results
    if rs is not None:
        section("Best Configuration", "🏆")
        c1, c2 = st.columns([1, 2])
        with c1:
            kpi("Best CV F1 (weighted)", f"{rs.best_score_*100:.2f}%", "good")
        with c2:
            st.markdown("**Best parameters**")
            st.json(rs.best_params_)

        section("Test-Set Metrics", "📈")
        metric_row(st.session_state.metrics)

        section("Confusion Matrix", "🔲")
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                 "Confusion Matrix — Randomized Best"))

        section("Classification Report", "📋")
        st.dataframe(pd.DataFrame(classification_report(
            st.session_state.y_test, y_pred, digits=4, output_dict=True
        )).transpose(), use_container_width=True)


# ---------------- FEATURE IMPORTANCE ----------------
elif page == "📊 Feature Importance":
    hero("📊 Feature Importance",
         "Which network features drive the model's attack predictions?",
         badge="Step 6a / 6")

    if st.session_state.model is None:
        st.warning("⚠️ Train a model first."); st.stop()

    model = st.session_state.model
    if not hasattr(model, "feature_importances_"):
        st.error("Model has no feature_importances_."); st.stop()

    imps = model.feature_importances_
    names = st.session_state.feature_names or [f"f{i}" for i in range(len(imps))]
    imp_df = (pd.DataFrame({"Feature": names, "Importance": imps})
              .sort_values("Importance", ascending=False).reset_index(drop=True))

    top_n = st.slider("Show top N features", 5, min(30, len(imp_df)),
                      min(15, len(imp_df)))

    section(f"Top {top_n} Feature Importances", "🏅")
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
    palette = sns.color_palette("Blues_r", top_n)
    sns.barplot(data=imp_df.head(top_n), x="Importance", y="Feature",
                ax=ax, palette=palette)
    for i, v in enumerate(imp_df.head(top_n)["Importance"].values):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9,
                fontweight="bold", color="#334155")
    ax.set_xlabel("Relative importance")
    ax.set_title(f"Top {top_n} Feature Importances", pad=12)
    st.pyplot(fig)

    with st.expander("📄 Full ranked list"):
        st.dataframe(imp_df, use_container_width=True)


# ---------------- PREDICT ----------------
elif page == "🔮 Predict":
    hero("🔮 Predict on New Data",
         "Score a CSV of network sessions against the trained model.",
         badge="Step 6b / 6")

    if st.session_state.model is None:
        st.warning("⚠️ Train a model first."); st.stop()

    up = st.file_uploader("Upload feature CSV", type=["csv"])
    if up is not None:
        new_df = pd.read_csv(up)
        if "session_id" in new_df.columns:
            new_df = new_df.drop("session_id", axis=1)

        le = LabelEncoder()
        for c in ["protocol_type", "browser_type", "encryption_used"]:
            if c in new_df.columns:
                new_df[c] = le.fit_transform(new_df[c].astype(str))

        expected = st.session_state.feature_names
        new_enc = pd.get_dummies(new_df).reindex(
            columns=expected, fill_value=0)

        X_scaled = st.session_state.scaler.transform(new_enc)

        preds = st.session_state.model.predict(X_scaled)
        probas = st.session_state.model.predict_proba(X_scaled)[:, 1]

        out = new_df.copy()
        out["prediction"] = preds
        out["attack_probability"] = probas.round(4)

        n_attacks = int((preds == 1).sum())
        n_safe = int((preds == 0).sum())

        section("Prediction Summary", "📊")
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Sessions Scored", f"{len(out):,}", "accent")
        with c2: kpi("Attacks Detected", f"{n_attacks:,}", "danger")
        with c3: kpi("Benign Sessions", f"{n_safe:,}", "good")

        section("Sample Predictions", "👀")
        st.dataframe(out.head(50), use_container_width=True)

        section("Prediction Distribution", "📈")
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = out["prediction"].value_counts()
        colors = ["#10b981" if v == 0 else "#ef4444" for v in vc.index]
        ax.bar(vc.index.astype(str), vc.values, color=colors,
               edgecolor="white", linewidth=1.5)
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
        ax.set_xlabel("Prediction (0 = benign, 1 = attack)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.download_button(
            "⬇️ Download Predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

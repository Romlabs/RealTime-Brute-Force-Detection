# -*- coding: utf-8 -*-
"""
Real-Time Brute-Force Detection via Predictive Machine Learning
Streamlit Deployable Version
"""

import os
import warnings
import numpy as np
import pandas as pd
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

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Brute-Force Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ------------------------------------------------------------------
# Session state helpers
# ------------------------------------------------------------------
def init_state():
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "X_res": None,
        "y_res": None,
        "feature_names": None,
        "model": None,
        "model_name": None,
        "metrics": None,
        "best_params": None,
        "grid_results": None,
        "rand_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("🛡️ Brute-Force Detection")
st.sidebar.markdown(
    "A machine-learning pipeline for detecting brute-force attacks "
    "in network sessions."
)

page = st.sidebar.radio(
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
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Random Forest · SMOTE")

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_from_kaggle():
    """Download the Kaggle dataset and return a DataFrame."""
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
        raise FileNotFoundError("No CSV file found in Kaggle dataset.")
    return pd.read_csv(csv_file)


def clean_dataframe(df: pd.DataFrame):
    """Apply label encoding + dedup. Returns the cleaned DataFrame."""
    df = df.copy()
    if "session_id" in df.columns:
        df = df.drop("session_id", axis=1)

    le = LabelEncoder()
    for col in ["protocol_type", "browser_type", "encryption_used"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_and_scale(df):
    """Split, one-hot encode, scale, and (optionally) SMOTE."""
    X = df.drop("attack_detected", axis=1)
    y = df["attack_detected"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    # align columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    feature_names = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return (
        X_train_s, X_test_s, y_train, y_test,
        feature_names, scaler,
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred
    )
    return {
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "F1 Score": f1_score(y_test, y_pred, zero_division=0) * 100,
        "PR-AUC": average_precision_score(y_test, y_proba) * 100,
    }, y_pred, y_proba


def plot_confusion(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return fig


# ==================================================================
# PAGES
# ==================================================================

# ---------------- Overview ----------------
if page == "🏠 Overview":
    st.title("🛡️ Real-Time Brute-Force Detection")
    st.markdown(
        """
        ### Predictive Machine Learning Pipeline

        This dashboard reproduces the full notebook pipeline for detecting
        **brute-force attacks** on network sessions:

        1. **Load** the *Cybersecurity Intrusion Detection* dataset.
        2. **Clean** the data (label encoding, deduplication).
        3. **Balance** the classes with **SMOTE**.
        4. **Train** a `RandomForestClassifier`.
        5. **Tune** hyperparameters with **Grid Search** and
           **Randomized Search**.
        6. **Visualize** metrics, confusion matrices, and feature
           importances.

        Use the sidebar to walk through each stage.
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Imbalance Handling", "SMOTE")
    col3.metric("Tuning", "Grid + Randomized")


# ---------------- Load Data ----------------
elif page == "📥 Load Data":
    st.title("📥 Load Dataset")

    source = st.radio(
        "Choose a data source:",
        ["Kaggle (auto-download)", "Upload CSV"],
        horizontal=True,
    )

    if source == "Kaggle (auto-download)":
        if st.button("⬇️ Download from Kaggle"):
            with st.spinner("Downloading dataset from Kaggle..."):
                try:
                    df = load_from_kaggle()
                    st.session_state.df_raw = df
                    st.success(f"Loaded {len(df):,} rows.")
                except Exception as e:
                    st.error(f"Failed to load: {e}")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state.df_raw = df
            st.success(f"Loaded {len(df):,} rows.")

    df = st.session_state.df_raw
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Duplicate rows", int(df.duplicated().sum()))

        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str).rename("dtype"))

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().rename("missing"))

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)


# ---------------- Cleaning ----------------
elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")

    if st.session_state.df_raw is None:
        st.warning("Please load a dataset first.")
        st.stop()

    df = st.session_state.df_raw

    st.markdown(
        """
        Cleaning steps:
        - Drop `session_id` (non-predictive identifier).
        - Label-encode `protocol_type`, `browser_type`, `encryption_used`.
        - Remove duplicate rows.
        """
    )

    if st.button("▶️ Run Cleaning"):
        with st.spinner("Cleaning..."):
            clean_df = clean_dataframe(df)
            st.session_state.df_clean = clean_df
        st.success(
            f"Cleaned: {len(df):,} → {len(clean_df):,} rows"
        )

    clean_df = st.session_state.df_clean
    if clean_df is not None:
        st.subheader("Cleaned Preview")
        st.dataframe(clean_df.head(10), use_container_width=True)

        if "attack_detected" in clean_df.columns:
            st.subheader("Target Distribution (before SMOTE)")
            vc = clean_df["attack_detected"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            vc.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
            ax.set_xlabel("attack_detected")
            ax.set_ylabel("Count")
            ax.set_title("Attack Distribution")
            st.pyplot(fig)

            pct = clean_df["attack_detected"].value_counts(normalize=True) * 100
            st.write("Percentage:")
            st.dataframe(pct.rename("percent").to_frame())

        st.subheader("Outlier Boxplot — session_duration")
        if "session_duration" in clean_df.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.boxplot(clean_df["session_duration"].dropna(), vert=False)
            ax.set_xlabel("session_duration")
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        num_df = clean_df.select_dtypes(include=[np.number])
        if num_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm",
                        fmt=".2f", ax=ax)
            st.pyplot(fig)


# ---------------- SMOTE ----------------
elif page == "⚖️ Class Imbalance (SMOTE)":
    st.title("⚖️ Class Imbalance & SMOTE")

    if st.session_state.df_clean is None:
        st.warning("Run cleaning first.")
        st.stop()

    df = st.session_state.df_clean.copy()

    if "attack_detected" not in df.columns:
        st.error("Column `attack_detected` not found.")
        st.stop()

    if st.button("▶️ Split & Apply SMOTE"):
        with st.spinner("Splitting and resampling..."):
            (X_train, X_test, y_train, y_test,
             feat_names, scaler) = split_and_scale(df)

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.X_res = X_res
            st.session_state.y_res = y_res
            st.session_state.feature_names = feat_names

        st.success("SMOTE applied.")

    X_res = st.session_state.X_res
    y_res = st.session_state.y_res
    y_train = st.session_state.y_train

    if y_res is not None:
        c1, c2 = st.columns(2)
        c1.metric("Train rows (before SMOTE)", len(y_train))
        c2.metric("Train rows (after SMOTE)", len(y_res))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before SMOTE")
            fig, ax = plt.subplots(figsize=(5, 4))
            y_train.value_counts().plot(kind="bar", ax=ax,
                                        color=["#4C72B0", "#DD8452"])
            ax.set_title("Before SMOTE")
            st.pyplot(fig)
        with col2:
            st.subheader("After SMOTE")
            fig, ax = plt.subplots(figsize=(5, 4))
            y_res.value_counts().plot(kind="bar", ax=ax,
                                      color=["#4C72B0", "#DD8452"])
            ax.set_title("After SMOTE")
            st.pyplot(fig)


# ---------------- Train ----------------
elif page == "🌲 Train Model":
    st.title("🌲 Train Random Forest")

    if st.session_state.X_res is None:
        st.warning("Run SMOTE step first.")
        st.stop()

    n_estimators = st.slider("n_estimators", 50, 500, 100, 50)
    max_depth = st.select_slider(
        "max_depth", options=[5, 10, 20, 30, None], value=None
    )
    use_balanced = st.checkbox("class_weight='balanced'", value=False)

    if st.button("▶️ Train Model"):
        with st.spinner("Training..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced" if use_balanced else None,
            )
            model.fit(st.session_state.X_res, st.session_state.y_res)
            metrics, y_pred, _ = evaluate_model(
                model, st.session_state.X_test, st.session_state.y_test
            )
            st.session_state.model = model
            st.session_state.model_name = "Random Forest"
            st.session_state.metrics = metrics

        st.success("Model trained.")

    if st.session_state.metrics is not None:
        st.subheader("Test-Set Metrics")
        m = st.session_state.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['Accuracy']:.2f}%")
        c2.metric("Precision", f"{m['Precision']:.2f}%")
        c3.metric("Recall", f"{m['Recall']:.2f}%")
        c4.metric("F1", f"{m['F1 Score']:.2f}%")
        c5.metric("PR-AUC", f"{m['PR-AUC']:.2f}%")

        y_pred = st.session_state.model.predict(st.session_state.X_test)
        fig = plot_confusion(
            st.session_state.y_test, y_pred,
            "Confusion Matrix — Random Forest",
        )
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(
            st.session_state.y_test, y_pred, digits=4, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())


# ---------------- Grid Search ----------------
elif page == "🔧 Grid Search":
    st.title("🔧 Grid Search Hyperparameter Tuning")

    if st.session_state.X_res is None:
        st.warning("Run SMOTE step first.")
        st.stop()

    st.markdown("Searches over a small grid (reduced for interactivity).")

    if st.button("▶️ Run Grid Search"):
        param_grid = {
            "n_estimators": [100, 200],
            "max_features": ["sqrt", "log2"],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        gs = GridSearchCV(
            rf, param_grid, cv=3, n_jobs=-1,
            scoring="accuracy", verbose=0,
        )
        with st.spinner("Running Grid Search (this can take a while)..."):
            gs.fit(st.session_state.X_res, st.session_state.y_res)

        st.session_state.grid_results = gs
        st.session_state.best_params = gs.best_params_

        best = gs.best_estimator_
        metrics, y_pred, _ = evaluate_model(
            best, st.session_state.X_test, st.session_state.y_test
        )
        st.session_state.model = best
        st.session_state.metrics = metrics
        st.session_state.model_name = "Random Forest (Grid Search)"

        st.success("Grid Search complete.")

    if st.session_state.grid_results is not None:
        gs = st.session_state.grid_results
        st.subheader("Best Parameters")
        st.json(gs.best_params_)
        st.write(f"**Best CV accuracy:** {gs.best_score_ * 100:.2f}%")

        st.subheader("All Grid Results")
        st.dataframe(pd.DataFrame(gs.cv_results_))

        m = st.session_state.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['Accuracy']:.2f}%")
        c2.metric("Precision", f"{m['Precision']:.2f}%")
        c3.metric("Recall", f"{m['Recall']:.2f}%")
        c4.metric("F1", f"{m['F1 Score']:.2f}%")
        c5.metric("PR-AUC", f"{m['PR-AUC']:.2f}%")

        y_pred = st.session_state.model.predict(st.session_state.X_test)
        fig = plot_confusion(
            st.session_state.y_test, y_pred,
            "Confusion Matrix — Grid Search Best",
        )
        st.pyplot(fig)


# ---------------- Randomized Search ----------------
elif page == "🎲 Randomized Search":
    st.title("🎲 Randomized Search Hyperparameter Tuning")

    if st.session_state.X_res is None:
        st.warning("Run SMOTE step first.")
        st.stop()

    n_iter = st.slider("n_iter", 5, 30, 10, 5)
    n_estimators = st.slider("Base n_estimators", 100, 600, 300, 50)

    if st.button("▶️ Run Randomized Search"):
        param_dist = {
            "max_features": ["sqrt", "log2", None],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": randint(2, 15),
            "min_samples_leaf": randint(1, 8),
            "bootstrap": [True, False],
            "class_weight": [None, "balanced"],
        }
        rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
        rs = RandomizedSearchCV(
            rf, param_dist, n_iter=n_iter, cv=3,
            scoring="f1_weighted", n_jobs=-1,
            random_state=42, verbose=0,
        )
        with st.spinner("Running Randomized Search..."):
            rs.fit(st.session_state.X_res, st.session_state.y_res)

        st.session_state.rand_results = rs

        best = rs.best_estimator_
        metrics, y_pred, _ = evaluate_model(
            best, st.session_state.X_test, st.session_state.y_test
        )
        st.session_state.model = best
        st.session_state.metrics = metrics
        st.session_state.model_name = "Random Forest (Randomized Search)"
        st.session_state.best_params = rs.best_params_

        st.success("Randomized Search complete.")

    if st.session_state.rand_results is not None:
        rs = st.session_state.rand_results
        st.subheader("Best Parameters")
        st.json(rs.best_params_)
        st.write(f"**Best CV F1 (weighted):** {rs.best_score_ * 100:.2f}%")

        m = st.session_state.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['Accuracy']:.2f}%")
        c2.metric("Precision", f"{m['Precision']:.2f}%")
        c3.metric("Recall", f"{m['Recall']:.2f}%")
        c4.metric("F1", f"{m['F1 Score']:.2f}%")
        c5.metric("PR-AUC", f"{m['PR-AUC']:.2f}%")

        y_pred = st.session_state.model.predict(st.session_state.X_test)
        fig = plot_confusion(
            st.session_state.y_test, y_pred,
            "Confusion Matrix — Randomized Search Best",
        )
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.dataframe(
            pd.DataFrame(
                classification_report(
                    st.session_state.y_test, y_pred,
                    digits=4, output_dict=True,
                )
            ).transpose()
        )


# ---------------- Feature Importance ----------------
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance")

    if st.session_state.model is None:
        st.warning("Train a model first.")
        st.stop()

    model = st.session_state.model
    if not hasattr(model, "feature_importances_"):
        st.error("Current model has no feature_importances_.")
        st.stop()

    importances = model.feature_importances_
    names = st.session_state.feature_names

    if names is None or len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]

    imp_df = (
        pd.DataFrame({"Feature": names, "Importance": importances})
        .sort_values("Importance", ascending=False)
    )

    st.subheader("Top Features")
    top_n = st.slider("Show top N", 5, min(30, len(imp_df)), min(15, len(imp_df)))
    top = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    sns.barplot(x="Importance", y="Feature", data=top, ax=ax,
                palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importances")
    st.pyplot(fig)

    st.dataframe(imp_df, use_container_width=True)


# ---------------- Predict ----------------
elif page == "🔮 Predict":
    st.title("🔮 Predict on New Data")

    if st.session_state.model is None:
        st.warning("Train a model first.")
        st.stop()

    st.markdown(
        "Upload a CSV containing the same feature columns used during "
        "training (minus `attack_detected`)."
    )

    uploaded = st.file_uploader("Upload feature CSV", type=["csv"])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)

        if "session_id" in new_df.columns:
            new_df = new_df.drop("session_id", axis=1)

        le = LabelEncoder()
        for col in ["protocol_type", "browser_type", "encryption_used"]:
            if col in new_df.columns:
                new_df[col] = le.fit_transform(new_df[col].astype(str))

        expected = st.session_state.feature_names
        new_enc = pd.get_dummies(new_df)
        new_enc = new_enc.reindex(columns=expected, fill_value=0)

        scaler = StandardScaler()
        # NOTE: Ideally reuse the fitted scaler; we refit for demo purposes.
        X_scaled = scaler.fit_transform(new_enc)

        preds = st.session_state.model.predict(X_scaled)
        probas = st.session_state.model.predict_proba(X_scaled)[:, 1]

        out = new_df.copy()
        out["prediction"] = preds
        out["attack_probability"] = probas.round(4)

        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

        st.subheader("Prediction Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        out["prediction"].value_counts().plot(kind="bar", ax=ax,
                                              color=["#4C72B0", "#DD8452"])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Count")
        st.pyplot(fig)

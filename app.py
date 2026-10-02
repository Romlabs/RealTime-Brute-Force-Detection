# -*- coding: utf-8 -*-
"""
Real-Time Brute-Force Detection via Predictive Machine Learning
Streamlit Cloud-safe version (fixed OOM, deprecated args, warnings).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend for Streamlit Cloud
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
# PAGE CONFIG
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
# SESSION STATE
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("🛡️ Brute-Force Detection")
st.sidebar.caption("Random Forest · SMOTE · Hyperparameter search")

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

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
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
        raise FileNotFoundError("No CSV found in Kaggle dataset.")
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
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else y_pred
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test), ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    return fig


def metric_row(m):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['Accuracy']:.2f}%")
    c2.metric("Precision", f"{m['Precision']:.2f}%")
    c3.metric("Recall",    f"{m['Recall']:.2f}%")
    c4.metric("F1",        f"{m['F1 Score']:.2f}%")
    c5.metric("PR-AUC",    f"{m['PR-AUC']:.2f}%")


# ==================================================================
# PAGES
# ==================================================================

if page == "🏠 Overview":
    st.title("🛡️ Real-Time Brute-Force Detection")
    st.markdown("""
    ### Predictive Machine Learning Pipeline
    1. **Load** the *Cybersecurity Intrusion Detection* dataset
    2. **Clean** (label encoding, dedup)
    3. **Balance** with **SMOTE**
    4. **Train** a Random Forest
    5. **Tune** via Grid / Randomized Search
    6. **Visualize** metrics, confusion matrix, feature importances

    > ⚙️ On Streamlit Cloud, hyperparameter searches use small grids
    > and `n_jobs=1` to fit the ~1 GB / 1 CPU container.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("Imbalance", "SMOTE")
    col3.metric("Tuning", "Grid + Randomized")


elif page == "📥 Load Data":
    st.title("📥 Load Dataset")
    source = st.radio("Source:",
                      ["Kaggle (auto-download)", "Upload CSV"],
                      horizontal=True)

    if source == "Kaggle (auto-download)":
        if st.button("⬇️ Download from Kaggle"):
            with st.spinner("Downloading..."):
                try:
                    df = load_from_kaggle()
                    st.session_state.df_raw = df
                    st.success(f"Loaded {len(df):,} rows.")
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            st.session_state.df_raw = pd.read_csv(up)
            st.success(f"Loaded {len(st.session_state.df_raw):,} rows.")

    df = st.session_state.df_raw
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(10), width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Cols", df.shape[1])
        c3.metric("Duplicates", int(df.duplicated().sum()))
        with st.expander("Data types / missing / describe"):
            st.write(df.dtypes)
            st.write(df.isnull().sum())
            st.dataframe(df.describe())


elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")
    if st.session_state.df_raw is None:
        st.warning("Load a dataset first."); st.stop()

    if st.button("▶️ Run Cleaning"):
        with st.spinner("Cleaning..."):
            st.session_state.df_clean = clean_dataframe(st.session_state.df_raw)
        st.success(f"Cleaned: {len(st.session_state.df_raw):,} → "
                   f"{len(st.session_state.df_clean):,}")

    df = st.session_state.df_clean
    if df is not None:
        st.dataframe(df.head(10), width="stretch")

        if "attack_detected" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Target distribution")
                fig, ax = plt.subplots()
                df["attack_detected"].value_counts().plot(kind="bar", ax=ax,
                    color=["#4C72B0", "#DD8452"])
                st.pyplot(fig)
            with col2:
                st.subheader("Correlation")
                num = df.select_dtypes(include=[np.number])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(num.corr(), annot=True, cmap="coolwarm",
                            fmt=".2f", ax=ax)
                st.pyplot(fig)

        if "session_duration" in df.columns:
            st.subheader("Outlier boxplot — session_duration")
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.boxplot(df["session_duration"].dropna(), vert=False)
            st.pyplot(fig)


elif page == "⚖️ Class Imbalance (SMOTE)":
    st.title("⚖️ Class Imbalance & SMOTE")
    if st.session_state.df_clean is None:
        st.warning("Run cleaning first."); st.stop()

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
        st.success("SMOTE done.")

    if st.session_state.y_res is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Before SMOTE")
            fig, ax = plt.subplots()
            st.session_state.y_train.value_counts().plot(kind="bar", ax=ax,
                color=["#4C72B0", "#DD8452"])
            st.pyplot(fig)
        with col2:
            st.subheader("After SMOTE")
            fig, ax = plt.subplots()
            st.session_state.y_res.value_counts().plot(kind="bar", ax=ax,
                color=["#4C72B0", "#DD8452"])
            st.pyplot(fig)


elif page == "🌲 Train Model":
    st.title("🌲 Train Random Forest")
    if st.session_state.X_res is None:
        st.warning("Run SMOTE first."); st.stop()

    n_estimators = st.slider("n_estimators", 50, 300, 100, 50)
    max_depth = st.select_slider("max_depth",
                                 options=[5, 10, 20, 30, None], value=None)
    balanced = st.checkbox("class_weight='balanced'", value=False)

    if st.button("▶️ Train"):
        with st.spinner("Training..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=42, n_jobs=1,      # 🔑 cloud-safe
                class_weight="balanced" if balanced else None,
            )
            model.fit(st.session_state.X_res, st.session_state.y_res)
            metrics, y_pred, _ = evaluate_model(
                model, st.session_state.X_test, st.session_state.y_test
            )
            st.session_state.model = model
            st.session_state.metrics = metrics
            st.session_state.model_name = "Random Forest"
        st.success("Trained.")

    if st.session_state.metrics:
        metric_row(st.session_state.metrics)
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                 "Confusion Matrix — Random Forest"))
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(classification_report(
            st.session_state.y_test, y_pred, digits=4, output_dict=True
        )).transpose())


elif page == "🔧 Grid Search":
    st.title("🔧 Grid Search (Cloud-safe)")
    if st.session_state.X_res is None:
        st.warning("Run SMOTE first."); st.stop()

    st.info("Reduced grid + cv=2 + n_jobs=1 to fit Streamlit Cloud.")

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
        st.success("Done.")

    gs = st.session_state.grid_results
    if gs is not None:
        st.subheader("Best params")
        st.json(gs.best_params_)
        st.write(f"**Best CV accuracy:** {gs.best_score_*100:.2f}%")
        metric_row(st.session_state.metrics)
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                 "Confusion — Grid Best"))


elif page == "🎲 Randomized Search":
    st.title("🎲 Randomized Search (Cloud-safe)")
    if st.session_state.X_res is None:
        st.warning("Run SMOTE first."); st.stop()

    st.info("n_iter capped at 20, cv=2, n_jobs=1.")

    n_iter = st.slider("n_iter", 5, 20, 10, 5)

    if st.button("▶️ Run Randomized Search"):
        param_dist = {
            "n_estimators":      [100],                 # keep small
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
        st.success("Done.")

    rs = st.session_state.rand_results
    if rs is not None:
        st.subheader("Best params")
        st.json(rs.best_params_)
        st.write(f"**Best CV F1:** {rs.best_score_*100:.2f}%")
        metric_row(st.session_state.metrics)
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        st.pyplot(plot_confusion(st.session_state.y_test, y_pred,
                                 "Confusion — Randomized Best"))
        st.dataframe(pd.DataFrame(classification_report(
            st.session_state.y_test, y_pred, digits=4, output_dict=True
        )).transpose())


elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance")
    if st.session_state.model is None:
        st.warning("Train a model first."); st.stop()

    model = st.session_state.model
    if not hasattr(model, "feature_importances_"):
        st.error("Model has no feature_importances_."); st.stop()

    imps = model.feature_importances_
    names = st.session_state.feature_names or \
            [f"f{i}" for i in range(len(imps))]
    imp_df = (pd.DataFrame({"Feature": names, "Importance": imps})
              .sort_values("Importance", ascending=False))

    top_n = st.slider("Top N", 5, min(30, len(imp_df)),
                      min(15, len(imp_df)))
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    sns.barplot(data=imp_df.head(top_n), x="Importance", y="Feature",
                ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importances")
    st.pyplot(fig)
    st.dataframe(imp_df, width="stretch")


elif page == "🔮 Predict":
    st.title("🔮 Predict on New Data")
    if st.session_state.model is None:
        st.warning("Train a model first."); st.stop()

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

        # ✅ reuse the scaler fitted during SMOTE step
        X_scaled = st.session_state.scaler.transform(new_enc)

        preds = st.session_state.model.predict(X_scaled)
        probas = st.session_state.model.predict_proba(X_scaled)[:, 1]

        out = new_df.copy()
        out["prediction"] = preds
        out["attack_probability"] = probas.round(4)

        st.success(f"Predicted {len(out)} rows.")
        st.dataframe(out.head(50), width="stretch")
        st.download_button("⬇️ Download Predictions",
                           out.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")
        fig, ax = plt.subplots()
        out["prediction"].value_counts().plot(kind="bar", ax=ax,
            color=["#4C72B0", "#DD8452"])
        st.pyplot(fig)

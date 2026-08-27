"""
Telco Customer Churn — Interactive Streamlit Dashboard
========================================================
Rebuilt from an exploratory notebook (TelcoChurn.ipynb) into a production-style
Streamlit app with four sections:

    1. Overview           – KPIs & headline churn numbers
    2. Explore the Data   – categorical / numerical drivers of churn
    3. Model Performance  – train & compare classifiers, ROC/PR curves
    4. Predict a Customer – single-customer & batch churn scoring

Run with:  streamlit run telco_churn_dashboard.py
"""

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Optional dependencies — the app degrades gracefully if these aren't installed
# --------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ==========================================================================
# PAGE CONFIG
# ==========================================================================
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#E94B3C"
SECONDARY = "#2D2926"
ACCENT = "#2ECC71"
COLOR_MAP = {"No": "#2563EB", "Yes": "#F97316", 0: "#2563EB", 1: "#F97316"}

st.markdown(
    """
    <style>
    .metric-card {padding: 0.5rem 0;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# DATA LOADING & CLEANING
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the notebook's cleaning: fix blank TotalCharges, drop ID."""
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Same fix as the notebook: fill blanks from the previous row,
        # falling back to the column median for any that remain.
        df["TotalCharges"] = df["TotalCharges"].ffill()
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(
            df["SeniorCitizen"]
        )

    return df


def get_feature_groups(df: pd.DataFrame, id_col="customerID", target_col="Churn"):
    cols = [c for c in df.columns if c not in (id_col, target_col)]
    categorical = [c for c in cols if df[c].nunique() <= 6 and df[c].dtype == object]
    numeric = [c for c in cols if c not in categorical]
    return categorical, numeric


# ==========================================================================
# ENCODING / MODELING HELPERS
# ==========================================================================
@st.cache_resource(show_spinner=False)
def encode_features(df: pd.DataFrame, categorical_cols, target_col="Churn"):
    """Label-encode categoricals + target, keep encoders for inverse lookups."""
    enc_df = df.copy()
    encoders = {}
    for col in categorical_cols + [target_col]:
        le = LabelEncoder()
        enc_df[col] = le.fit_transform(enc_df[col].astype(str))
        encoders[col] = le
    return enc_df, encoders


def build_model_zoo(selected_names):
    """Instantiate the requested classifiers (only the ones available)."""
    zoo = {}
    if "Logistic Regression" in selected_names:
        zoo["Logistic Regression"] = LogisticRegression(max_iter=2000)
    if "Decision Tree" in selected_names:
        zoo["Decision Tree"] = DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=2, random_state=42
        )
    if "Random Forest" in selected_names:
        zoo["Random Forest"] = RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
        )
    if "Gradient Boosting" in selected_names:
        zoo["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
    if "XGBoost" in selected_names and XGB_AVAILABLE:
        zoo["XGBoost"] = XGBClassifier(
            learning_rate=0.05,
            max_depth=4,
            n_estimators=400,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
    if "LightGBM" in selected_names and LGBM_AVAILABLE:
        zoo["LightGBM"] = LGBMClassifier(
            learning_rate=0.05, max_depth=4, n_estimators=400, verbosity=-1
        )

    if "Stacking Ensemble" in selected_names and len(zoo) >= 2:
        base_estimators = [(n, m) for n, m in zoo.items() if n != "Logistic Regression"]
        if len(base_estimators) >= 2:
            zoo["Stacking Ensemble"] = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=2000),
                n_jobs=-1,
            )
    return zoo


@st.cache_resource(show_spinner=True)
def train_and_evaluate(
    _X_train, _y_train, _X_test, _y_test, model_names, use_smote, data_key
):
    """Train every requested model once; cached on (data, model list, smote)."""
    X_train, y_train = _X_train.copy(), _y_train.copy()

    if use_smote and SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    zoo = build_model_zoo(model_names)
    fitted = {}
    rows = []

    for name, clf in zoo.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(_X_test)
        y_prob = clf.predict_proba(_X_test)[:, 1]

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(_y_test, y_pred),
                "Precision": precision_score(_y_test, y_pred, zero_division=0),
                "Recall": recall_score(_y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(_y_test, y_pred, zero_division=0),
                "ROC AUC": roc_auc_score(_y_test, y_prob),
            }
        )
        fitted[name] = clf

    results_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(
        drop=True
    )
    return fitted, results_df


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return None
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================================
# SIDEBAR — DATA SOURCE & CONFIG
# ==========================================================================
st.sidebar.title("📡 Telco Churn Dashboard")
st.sidebar.caption("Upload the WA_Fn-UseC_-Telco-Customer-Churn.csv dataset (or a similar telecom churn export) to begin.")

uploaded = st.sidebar.file_uploader("Customer churn CSV", type=["csv"])

if uploaded is None:
    st.title("📡 Telco Customer Churn Dashboard")
    st.info(
        "👈 Upload a customer churn CSV in the sidebar to get started. "
        "The app expects a `Churn` column (Yes/No) and typical telco fields "
        "like `tenure`, `MonthlyCharges`, `Contract`, `InternetService`, etc."
    )
    st.stop()

raw_df = load_data(uploaded)
df = clean_data(raw_df)

if "Churn" not in df.columns:
    st.error("This file doesn't have a `Churn` column — please upload the correct dataset.")
    st.stop()

id_col = "customerID" if "customerID" in df.columns else None
categorical_cols, numeric_cols = get_feature_groups(df, id_col=id_col)

st.sidebar.success(f"Loaded {len(df):,} customers · {df['Churn'].eq('Yes').mean():.1%} churn rate")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Settings")
default_models = ["Random Forest", "Gradient Boosting", "Decision Tree"]
if XGB_AVAILABLE:
    default_models.append("XGBoost")
if LGBM_AVAILABLE:
    default_models.append("LightGBM")

model_options = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
if XGB_AVAILABLE:
    model_options.append("XGBoost")
if LGBM_AVAILABLE:
    model_options.append("LightGBM")
model_options.append("Stacking Ensemble")

selected_models = st.sidebar.multiselect(
    "Models to train & compare", model_options, default=default_models
)
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
use_smote = st.sidebar.checkbox(
    "Balance classes with SMOTE",
    value=SMOTE_AVAILABLE,
    disabled=not SMOTE_AVAILABLE,
    help=None if SMOTE_AVAILABLE else "Install `imbalanced-learn` to enable SMOTE.",
)
random_state = 42

if not SMOTE_AVAILABLE:
    st.sidebar.caption("ℹ️ `imbalanced-learn` not installed — SMOTE disabled.")
if not XGB_AVAILABLE:
    st.sidebar.caption("ℹ️ `xgboost` not installed — option hidden.")
if not LGBM_AVAILABLE:
    st.sidebar.caption("ℹ️ `lightgbm` not installed — option hidden.")

# --------------------------------------------------------------------------
# Prepare model-ready data (shared by Model Performance & Prediction tabs)
# --------------------------------------------------------------------------
model_feature_cols = categorical_cols + numeric_cols
enc_df, encoders = encode_features(df[model_feature_cols + ["Churn"]], categorical_cols)

scaler = StandardScaler()
scale_cols = [c for c in numeric_cols if enc_df[c].nunique() > 6]
enc_df_scaled = enc_df.copy()
if scale_cols:
    enc_df_scaled[scale_cols] = scaler.fit_transform(enc_df_scaled[scale_cols])

X = enc_df_scaled[model_feature_cols]
y = enc_df_scaled["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=random_state
)

data_key = f"{uploaded.name}-{len(df)}-{test_size}-{use_smote}"


# ==========================================================================
# TABS
# ==========================================================================
tab_overview, tab_explore, tab_models, tab_predict = st.tabs(
    ["📊 Overview", "🔍 Explore the Data", "🤖 Model Performance", "🎯 Predict a Customer"]
)

# --------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# --------------------------------------------------------------------------
with tab_overview:
    st.header("Business Overview")

    churn_rate = df["Churn"].eq("Yes").mean()
    avg_tenure = df["tenure"].mean() if "tenure" in df else np.nan
    avg_monthly = df["MonthlyCharges"].mean() if "MonthlyCharges" in df else np.nan
    total_revenue_at_risk = (
        df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()
        if "MonthlyCharges" in df
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{churn_rate:.1%}")
    c3.metric("Avg. Tenure", f"{avg_tenure:.1f} mo" if pd.notna(avg_tenure) else "—")
    c4.metric(
        "Monthly Revenue at Risk",
        f"${total_revenue_at_risk:,.0f}" if pd.notna(total_revenue_at_risk) else "—",
    )

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Customers"]
        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            hole=0.5,
            color="Churn",
            color_discrete_map=COLOR_MAP,
            title="Churn vs. Retained",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "Contract" in df.columns:
            grp = (
                df.groupby(["Contract", "Churn"]).size().reset_index(name="Customers")
            )
            fig = px.bar(
                grp,
                x="Contract",
                y="Customers",
                color="Churn",
                barmode="group",
                color_discrete_map=COLOR_MAP,
                title="Churn by Contract Type",
            )
            st.plotly_chart(fig, use_container_width=True)

    if "tenure" in df.columns:
        st.subheader("Tenure Distribution by Churn")
        fig = px.histogram(
            df,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=40,
            color_discrete_map=COLOR_MAP,
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 2 — EXPLORE THE DATA
# --------------------------------------------------------------------------
with tab_explore:
    st.header("What Drives Churn?")

    st.subheader("Categorical Features vs. Churn")
    cat_choice = st.multiselect(
        "Pick categorical features to inspect",
        [c for c in categorical_cols],
        default=[c for c in categorical_cols if c in ["Contract", "InternetService", "PaymentMethod"]][:3]
        or categorical_cols[:3],
    )
    n_cols = 2
    cols = st.columns(n_cols)
    for i, feat in enumerate(cat_choice):
        grp = df.groupby([feat, "Churn"]).size().reset_index(name="Customers")
        fig = px.bar(
            grp,
            x=feat,
            y="Customers",
            color="Churn",
            barmode="group",
            color_discrete_map=COLOR_MAP,
            title=f"{feat} vs. Churn",
        )
        cols[i % n_cols].plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Numerical Features vs. Churn")
    num_choice = st.selectbox("Pick a numeric feature", numeric_cols)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df,
            x=num_choice,
            color="Churn",
            barmode="overlay",
            nbins=40,
            color_discrete_map=COLOR_MAP,
            opacity=0.7,
            title=f"Distribution of {num_choice}",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(
            df,
            x="Churn",
            y=num_choice,
            color="Churn",
            color_discrete_map=COLOR_MAP,
            title=f"{num_choice} by Churn",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation with Churn")
    corr_df = enc_df.corr(numeric_only=True)["Churn"].drop("Churn").sort_values()
    fig = px.bar(
        corr_df,
        orientation="h",
        title="Feature Correlation with Churn (encoded)",
        color=corr_df.values,
        color_continuous_scale="RdBu_r",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, yaxis_title="", xaxis_title="Correlation")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 3 — MODEL PERFORMANCE
# --------------------------------------------------------------------------
with tab_models:
    st.header("Train & Compare Models")

    if not selected_models:
        st.warning("Select at least one model in the sidebar to train.")
        st.stop()

    if st.button("🚀 Train Models", type="primary"):
        st.session_state["trained"] = train_and_evaluate(
            X_train, y_train, X_test, y_test, tuple(selected_models), use_smote, data_key
        )

    if "trained" not in st.session_state:
        st.info("Click **Train Models** to fit the selected classifiers on this dataset.")
        st.stop()

    fitted_models, results_df = st.session_state["trained"]

    st.subheader("Metrics Comparison")
    st.dataframe(
        results_df.style.format(
            {c: "{:.3f}" for c in results_df.columns if c != "Model"}
        ).background_gradient(cmap="Greens", subset=["F1 Score", "ROC AUC"]),
        use_container_width=True,
    )

    best_model_name = results_df.iloc[0]["Model"]
    st.success(f"🏆 Best model by F1 score: **{best_model_name}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curves")
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        for name, clf in fitted_models.items():
            y_prob = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.2f})"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confusion Matrix")
        model_for_cm = st.selectbox("Model", list(fitted_models.keys()), key="cm_model")
        clf = fitted_models[model_for_cm]
        cm = confusion_matrix(y_test, clf.predict(X_test))
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Importance")
    model_for_imp = st.selectbox("Model", list(fitted_models.keys()), key="imp_model")
    imp_df = get_feature_importance(fitted_models[model_for_imp], model_feature_cols)
    if imp_df is not None:
        fig = px.bar(
            imp_df.head(15).sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Oranges",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(f"{model_for_imp} doesn't expose feature importances directly.")


# --------------------------------------------------------------------------
# TAB 4 — PREDICT A CUSTOMER
# --------------------------------------------------------------------------
with tab_predict:
    st.header("Score a Customer")

    if "trained" not in st.session_state:
        st.warning("Train a model in the **Model Performance** tab first.")
        st.stop()

    fitted_models, results_df = st.session_state["trained"]
    predict_model_name = st.selectbox(
        "Model to use for scoring", list(fitted_models.keys()), index=0
    )
    clf = fitted_models[predict_model_name]

    sub_tab_single, sub_tab_batch = st.tabs(["👤 Single Customer", "📁 Batch Upload"])

    # ---- Single customer form ----
    with sub_tab_single:
        st.caption("Fill in the customer's attributes, then score their churn risk.")
        input_data = {}
        form_cols = st.columns(3)

        for i, col in enumerate(categorical_cols):
            options = sorted(df[col].dropna().unique().tolist())
            input_data[col] = form_cols[i % 3].selectbox(col, options, key=f"in_{col}")

        for i, col in enumerate(numeric_cols):
            col_min, col_max = float(df[col].min()), float(df[col].max())
            col_mean = float(df[col].mean())
            input_data[col] = form_cols[(i + len(categorical_cols)) % 3].number_input(
                col, min_value=col_min, max_value=col_max, value=col_mean, key=f"in_{col}"
            )

        if st.button("🔮 Predict Churn Risk", type="primary"):
            input_df = pd.DataFrame([input_data])
            for col in categorical_cols:
                le = encoders[col]
                input_df[col] = le.transform(input_df[col].astype(str))
            if scale_cols:
                input_df[scale_cols] = scaler.transform(input_df[scale_cols])
            input_df = input_df[model_feature_cols]

            proba = clf.predict_proba(input_df)[0, 1]
            pred = "Yes" if proba >= 0.5 else "No"

            risk_color = "#F97316" if pred == "Yes" else "#2ECC71"
            g1, g2 = st.columns([1, 2])
            with g1:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        title={"text": "Churn Probability (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": risk_color},
                            "steps": [
                                {"range": [0, 40], "color": "#E8F8F0"},
                                {"range": [40, 70], "color": "#FEF3E2"},
                                {"range": [70, 100], "color": "#FDEAE8"},
                            ],
                        },
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                st.metric("Predicted Outcome", f"{'⚠️ Likely to Churn' if pred == 'Yes' else '✅ Likely to Stay'}")
                imp_df = get_feature_importance(clf, model_feature_cols)
                if imp_df is not None:
                    st.markdown("**Top factors this model relies on most (global importance):**")
                    st.dataframe(imp_df.head(5), use_container_width=True, hide_index=True)
                st.caption(
                    "Importance is the model's overall feature ranking, not a per-customer "
                    "explanation. For customer-level reasoning, use SHAP if installed."
                )

    # ---- Batch upload ----
    with sub_tab_batch:
        st.caption("Upload a CSV of customers (same columns as the training data, no `Churn` needed) to score them all at once.")
        batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch_upload")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            batch_clean = clean_data(batch_df)
            missing = [c for c in model_feature_cols if c not in batch_clean.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                enc_batch = batch_clean[model_feature_cols].copy()
                for col in categorical_cols:
                    le = encoders[col]
                    enc_batch[col] = enc_batch[col].astype(str).map(
                        lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
                    )
                if scale_cols:
                    enc_batch[scale_cols] = scaler.transform(enc_batch[scale_cols])

                probs = clf.predict_proba(enc_batch)[:, 1]
                out = batch_clean.copy()
                out["Churn_Probability"] = probs
                out["Churn_Prediction"] = np.where(probs >= 0.5, "Yes", "No")

                st.dataframe(
                    out.sort_values("Churn_Probability", ascending=False),
                    use_container_width=True,
                )

                csv_buffer = io.StringIO()
                out.to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇️ Download Scored Customers",
                    data=csv_buffer.getvalue(),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

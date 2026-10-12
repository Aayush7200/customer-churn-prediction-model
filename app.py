
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

model = load_model()

st.title("📊 Customer Churn Prediction")
st.write("Upload customer data to predict potential churn.")

uploaded_file = st.file_uploader(
    "Upload customer CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Create engineered features if needed
    if "dropped_call_rate" not in df.columns:
        if {"drop_vce_Mean", "plcd_vce_Mean"}.issubset(df.columns):
            df["dropped_call_rate"] = (
                df["drop_vce_Mean"] /
                (df["plcd_vce_Mean"] + 1)
            )

    if "overage_ratio" not in df.columns:
        if {"ovrrev_Mean", "rev_Mean"}.issubset(df.columns):
            df["overage_ratio"] = (
                df["ovrrev_Mean"] /
                (df["rev_Mean"] + 1)
            )

    if "revenue_per_minute" not in df.columns:
        if {"rev_Mean", "mou_Mean"}.issubset(df.columns):
            df["revenue_per_minute"] = (
                df["rev_Mean"] /
                (df["mou_Mean"] + 1)
            )

    # Remove target and identifier
    X = df.drop(
        columns=["churn", "Customer_ID"],
        errors="ignore"
    )

    # Check that the model's expected columns exist
    expected = list(
        model.named_steps["preprocessor"].feature_names_in_
    )

    missing = [col for col in expected if col not in X.columns]

    if missing:
        st.error(
            "Your CSV is missing required columns: "
            + ", ".join(missing)
        )
    else:
        X = X[expected]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        results = pd.DataFrame({
            "Prediction": [
                "Churn" if p == 1 else "No Churn"
                for p in predictions
            ],
            "Churn Probability (%)": (
                probabilities * 100
            ).round(2)
        })

        st.subheader("Prediction Results")

        total = len(results)
        churn_count = (predictions == 1).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total)
        col2.metric("Predicted Churn", int(churn_count))
        col3.metric(
            "Predicted Churn Rate",
            f"{churn_count / total * 100:.2f}%"
        )

        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
        
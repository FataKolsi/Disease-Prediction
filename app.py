import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Model and Data
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("random_forest.joblib")  # Random Forest or Decision Tree model
    return model

@st.cache_data
def load_symptoms():
    # Load the training dataset to get column names
    df = pd.read_csv("training_dataset.csv")  
    symptom_cols = df.columns[1:]  # first column is 'disease'
    return symptom_cols, df

clf = load_model()
symptom_cols, df_train = load_symptoms()

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="Disease Prediction App", page_icon="🧠", layout="wide")

st.title("🧠 Disease Prediction App")
st.markdown("""
Select the symptoms you are experiencing below, and the model will predict the most likely diseases ranked by probability.
""")

# Sidebar for user input
st.sidebar.header("User Input")
selected_symptoms = st.sidebar.multiselect(
    "Select your symptoms:",
    options=symptom_cols,
    help="Check all symptoms that apply."
)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.sidebar.button("Predict"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Create input vector (1 for selected symptoms, 0 otherwise)
        input_data = np.zeros(len(symptom_cols))
        for symptom in selected_symptoms:
            if symptom in symptom_cols:
                idx = np.where(symptom_cols == symptom)[0][0]
                input_data[idx] = 1

        input_df = pd.DataFrame([input_data], columns=symptom_cols)

        try:
            # Predict probabilities if supported
            proba = clf.predict_proba(input_df)[0]
            diseases = clf.classes_

            results = pd.DataFrame({
                "Disease": diseases,
                "Probability": proba
            }).sort_values(by="Probability", ascending=False)

            # Normalize probabilities for cleaner display
            results["Probability"] = results["Probability"] / results["Probability"].sum()

            st.subheader("Top 5 Predicted Diseases")
            st.table(results.head(5).style.format({"Probability": "{:.2%}"}))

        except AttributeError:
            # For models without predict_proba (shouldn't happen with RandomForest)
            pred = clf.predict(input_df)[0]
            st.subheader("Predicted Disease")
            st.success(f"🩺 {pred}")
else:
    st.info("👈 Select symptoms and click **Predict** to get results.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")


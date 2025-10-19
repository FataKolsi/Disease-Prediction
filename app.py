import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Model and Symptom Metadata
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")  # Your pre-trained model
    return model

@st.cache_data
def load_symptoms():
    # Must match the columns used in training (One-Hot Encoding order)
    df = pd.read_csv("training_dataset.csv")  
    symptom_cols = df.columns[1:]  # first column = disease
    return symptom_cols, df

clf_dt = load_model()
symptom_cols, df_train = load_symptoms()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Disease Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Disease Prediction App")
st.markdown("Select your symptoms below to get the most likely diseases ranked by probability.")

# Sidebar
st.sidebar.header("User Input")
selected_symptoms = st.sidebar.multiselect(
    "Select symptoms you are experiencing:",
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
        # Create input vector
        input_data = np.zeros(len(symptom_cols))
        for symptom in selected_symptoms:
            if symptom in symptom_cols:
                input_data[np.where(symptom_cols == symptom)[0][0]] = 1
        
        input_df = pd.DataFrame([input_data], columns=symptom_cols)
        
        # Predict probabilities
        try:
            proba = clf_dt.predict_proba(input_df)[0]
            diseases = clf_dt.classes_
            results = pd.DataFrame({
                "Disease": diseases,
                "Probability": proba
            }).sort_values(by="Probability", ascending=False)

            st.subheader("Top 5 Predicted Diseases")
            st.table(results.head(5).style.format({"Probability": "{:.2%}"}))
            
            st.bar_chart(results.head(5).set_index("Disease"))

        except AttributeError:
            # If model does not support predict_proba
            pred = clf_dt.predict(input_df)[0]
            st.subheader("Predicted Disease")
            st.success(f"🩺 {pred}")
else:
    st.info("👈 Select symptoms and click **Predict** to get results.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(layout="wide")

# ---------------- HEADER ---------------- #
st.markdown("""
<h1 style='text-align:center;'>💡 Insurance Premium Predictor</h1>
<hr>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.header("User Profile")

    age = st.number_input("Age", 18, 100, 30)
    height = st.number_input("Height (cm)", 100, 220, 160)
    weight = st.number_input("Weight (kg)", 30, 150, 60)

    height_m = height / 100
    bmi = weight / (height_m ** 2)

    # BMI category + tips
    if bmi < 18.5:
        bmi_category = "Underweight"
        color = "#3498db"
        tips = [
            "Eat energy-dense foods",
            "Focus on strength training",
            "Add snacks between meals",
            "Ensure vitamin intake",
            "Maintain oral hygiene"
        ]
    elif bmi < 25:
        bmi_category = "Normal"
        color = "#2ecc71"
        tips = [
            "Follow balanced diet",
            "Mix cardio & strength",
            "Sleep well",
            "Stay hydrated",
            "Maintain hygiene"
        ]
    elif bmi < 30:
        bmi_category = "Overweight"
        color = "#f39c12"
        tips = [
            "Control portions",
            "Do low-impact cardio",
            "Manage stress eating",
            "Increase movement",
            "Use good footwear"
        ]
    else:
        bmi_category = "Obese"
        color = "#e74c3c"
        tips = [
            "Increase fiber intake",
            "Focus on mobility",
            "Track health metrics",
            "Regular checkups",
            "Maintain skin hygiene"
        ]

    st.markdown(f"""
    <div style='background-color:{color};padding:12px;border-radius:10px;color:white'>
    <b>BMI:</b> {bmi:.2f} <br>
    <b>Category:</b> {bmi_category}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Health Tips")
    for t in tips:
        st.write("•", t)

# ---------------- MEDICAL INPUTS ---------------- #
st.markdown("### 🩺 Medical Inputs")

c1, c2 = st.columns(2)

with c1:
    bp = st.selectbox("Blood Pressure", [0,1])
    diabetes = st.selectbox("Diabetes", [0,1])
    transplant = st.selectbox("Transplants", [0,1])

with c2:
    chronic = st.selectbox("Chronic Diseases", [0,1])
    cancer = st.selectbox("Cancer History", [0,1])
    allergies = st.selectbox("Allergies", [0,1])
    surgeries = st.selectbox("Surgeries", [0,1,2,3])

# ---------------- GAUGE FUNCTION ---------------- #
def show_gauge(pred):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Risk Meter"},
        gauge={
            'axis': {'range': [0, 50000]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 20000], 'color': "green"},
                {'range': [20000, 35000], 'color': "orange"},
                {'range': [35000, 50000], 'color': "red"}
            ],
        }
    ))
    return fig

# ---------------- PREDICTION ---------------- #
st.markdown("---")

if st.button("🚀 Predict Premium"):

    data = np.array([[age, height, weight, bp, diabetes,
                      transplant, chronic, cancer, surgeries]])

    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]

    # Risk classification
    if pred < 20000:
        risk = "Low Risk"
        risk_color = "#2ecc71"
    elif pred < 35000:
        risk = "Medium Risk"
        risk_color = "#f39c12"
    else:
        risk = "High Risk"
        risk_color = "#e74c3c"

    # ---------------- CARDS ---------------- #
    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"""
        <div style='background:#1f2937;padding:20px;border-radius:12px;color:white'>
        <h3>💰 Premium</h3>
        <h1>₹ {int(pred)}</h1>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div style='background:{risk_color};padding:20px;border-radius:12px;color:white'>
        <h3>⚠️ Risk</h3>
        <h1>{risk}</h1>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- GAUGE ---------------- #
    st.markdown("### 🔥 Risk Meter")
    st.plotly_chart(show_gauge(pred), use_container_width=True)

    # ---------------- CHART ---------------- #
    st.markdown("### 📊 Premium vs Risk Levels")

    categories = ["Low", "Medium", "High"]
    values = [20000, 35000, 50000]

    fig = plt.figure()
    plt.bar(categories, values)
    plt.axhline(pred)
    plt.title("Premium vs Risk")

    st.pyplot(fig)

    # ---------------- INSIGHTS ---------------- #
    st.markdown("### 📌 Insights")

    if risk == "Low Risk":
        st.success("Healthy premium zone. Maintain your lifestyle.")
    elif risk == "Medium Risk":
        st.warning("Moderate risk. Improving lifestyle can reduce premium.")
    else:
        st.error("High risk. Focus on health improvements.")
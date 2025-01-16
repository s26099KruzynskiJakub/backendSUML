import streamlit as st
import requests
import pickle

# Adres serwera Flask
FLASK_API_URL = 'http://localhost:5000/predict'

# Nagłówki aplikacji Streamlit
st.title("Heart Attack Prediction")
st.write("This application predicts the likelihood of a heart attack based on various health factors.")

# Wczytaj słownik opcji z pliku dictionary_stripped.pkl
try:
    with open('dictionary_stripped.pkl', 'rb') as file:
        options_dict = pickle.load(file)
    st.sidebar.success("Loaded options successfully!")
except Exception as e:
    st.sidebar.error(f"Failed to load options: {e}")
    options_dict = {}

# Placeholder na wynik (na samej górze)
result_placeholder = st.empty()

# Form to collect user inputs
with st.form(key='health_form'):
    sex = st.selectbox("Sex", options_dict.get("Sex", ["Female", "Male"]))
    general_health = st.selectbox("General Health", options_dict.get("GeneralHealth", ["Very good", "Good", "Other"]))
    sleep_hours = st.number_input("Sleep Hours", min_value=1.0, max_value=24.0, step=0.1)
    removed_teeth = st.selectbox("Removed Teeth", options_dict.get("RemovedTeeth", ["None of them", "1 to 5", "Other"]))
    had_angina = st.selectbox("Had Angina", options_dict.get("HadAngina", ["Yes", "No"]))
    had_stroke = st.selectbox("Had Stroke", options_dict.get("HadStroke", ["Yes", "No"]))
    had_copd = st.selectbox("Had COPD", options_dict.get("HadCOPD", ["Yes", "No"]))
    had_diabetes = st.selectbox("Had Diabetes", options_dict.get("HadDiabetes", ["Yes", "No"]))
    difficulty_walking = st.selectbox("Difficulty Walking", options_dict.get("DifficultyWalking", ["Yes", "No"]))
    smoker_status = st.selectbox("Smoker Status", options_dict.get("SmokerStatus", ["Former smoker", "Current smoker", "Never smoked"]))
    chest_scan = st.selectbox("Chest Scan", options_dict.get("ChestScan", ["Yes", "No"]))
    age_category = st.selectbox("Age Category", options_dict.get("AgeCategory", ["Under 18", "18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 to 64", "65 to 69", "70 or above"]))
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
    alcohol_drinkers = st.selectbox("Alcohol Drinkers", options_dict.get("AlcoholDrinkers", ["Yes", "No"]))

    submit_button = st.form_submit_button(label='Predict Heart Attack')

# When the form is submitted, send the request to the Flask API
if submit_button:
    # Prepare data for API
    input_data = {
        "Sex": sex,
        "GeneralHealth": general_health,
        "SleepHours": sleep_hours,
        "RemovedTeeth": removed_teeth,
        "HadAngina": had_angina,
        "HadStroke": had_stroke,
        "HadCOPD": had_copd,
        "HadDiabetes": had_diabetes,
        "DifficultyWalking": difficulty_walking,
        "SmokerStatus": smoker_status,
        "ChestScan": chest_scan,
        "AgeCategory": age_category,
        "BMI": bmi,
        "AlcoholDrinkers": alcohol_drinkers
    }

    try:
        response = requests.post(FLASK_API_URL, json=input_data)

        # Log response
        st.sidebar.text(f"Response Status Code: {response.status_code}")
        st.sidebar.json(response.json())

        if response.status_code == 200:
            prediction = response.json().get('prediction')

            # Display result at the top with a colored box
            if prediction == "Yes":
                result_placeholder.markdown(
                    '<div style="background-color:#FFCCCC;padding:20px;border-radius:10px;text-align:center;">'
                    '<h1 style="color:#CC0000;">Prediction: Heart Attack Likely</h1>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                result_placeholder.markdown(
                    '<div style="background-color:#CCFFCC;padding:20px;border-radius:10px;text-align:center;">'
                    '<h1 style="color:#008800;">Prediction: Heart Attack Unlikely</h1>'
                    '</div>',
                    unsafe_allow_html=True
                )
        else:
            result_placeholder.markdown(
                '<div style="background-color:#FFFFCC;padding:20px;border-radius:10px;text-align:center;">'
                '<h1 style="color:#CC9900;">Error in prediction. Please try again!</h1>'
                '</div>',
                unsafe_allow_html=True
            )
    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")
"""
Plik: main.py

Ten moduł prezentuje prosty serwer Flask przyjmujący dane (bez kolumny 'HadHeartAttack'),
wywołujący cztery modele (model_top_1, model_top_2, model_top_3, model_top_4) i zwracający
ostateczną etykietę "Yes" lub "No" na podstawie średniej ważonej ich prognoz.

Modele (GBC, LR, Ada, Ridge) zostały wytrenowane przy użyciu PyCaret 3.3.2 i scikit-learn 1.4.2.
"""

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Wczytanie zapisanych modeli (pliki .pkl)
print("JESTEM1")
model_top_1 = joblib.load("model_top_1.pkl")  # GBC
model_top_2 = joblib.load("model_top_2.pkl")  # LR
model_top_3 = joblib.load("model_top_3.pkl")  # Ada
model_top_4 = joblib.load("model_top_4.pkl")  # Ridge

# Wagi dla poszczególnych modeli (od najlepszego do najsłabszego)
WEIGHTS = {
    "model_top_1": 0.2503,  # GBC
    "model_top_2": 0.2502,  # LR
    "model_top_3": 0.2502,  # Ada
    "model_top_4": 0.2493  # Ridge
}

# Nazwy kolumn, które były wykorzystywane podczas trenowania (bez kolumny 'HadHeartAttack')
COLUMNS = [
    "Sex",
    "GeneralHealth",
    "SleepHours",
    "RemovedTeeth",
    "HadAngina",
    "HadStroke",
    "HadCOPD",
    "HadDiabetes",
    "DifficultyWalking",
    "SmokerStatus",
    "ChestScan",
    "AgeCategory",
    "BMI",
    "AlcoholDrinkers"
]


@app.route('/predict', methods=['POST'])
def predict_heart_attack():
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    """
    Endpoint: /predict
    ------------------
    Oczekuje w żądaniu POST danych wejściowych w formacie JSON.

    Przykładowe wywołanie:
    POST http://localhost:5000/predict
    Content-Type: application/json

    {
      "Sex": "Female",
      "GeneralHealth": "Very good",
      "SleepHours": 9.0,
      "RemovedTeeth": "None of them",
      "HadAngina": "No",
      "HadStroke": "No",
      "HadCOPD": "No",
      "HadDiabetes": "No",
      "DifficultyWalking": "No",
      "SmokerStatus": "Former smoker",
      "ChestScan": "No",
      "AgeCategory": "Age 65 to 69",
      "BMI": 27.99,
      "AlcoholDrinkers": "No"
    }

    Zwraca JSON z kluczem 'prediction' wskazującym, czy (Yes/No) wystąpi
    zdarzenie 'HadHeartAttack' (w interpretacji modeli).
    """
    # Pobranie danych z żądania
    data = request.get_json()

    # Stworzenie ramki danych z jednej obserwacji
    # (zakładamy, że klucze w data są zgodne z listą COLUMNS)
    df_input = pd.DataFrame([data], columns=COLUMNS)

    # Każdy z modeli zwraca (dla uproszczenia) 0 lub 1, ale
    # możesz również użyć predict_proba i brać kolumnę [:, 1] jeśli to modele binarne.
    # pred1 = model_top_1.predict(df_input)[0]
    # pred2 = model_top_2.predict(df_input)[0]
    # pred3 = model_top_3.predict(df_input)[0]
    # pred4 = model_top_4.predict(df_input)[0]
    pred1 = 1 if model_top_1.predict(df_input)[0] == "Yes" else 0
    pred2 = 1 if model_top_2.predict(df_input)[0] == "Yes" else 0
    pred3 = 1 if model_top_3.predict(df_input)[0] == "Yes" else 0
    pred4 = 1 if model_top_4.predict(df_input)[0] == "Yes" else 0

    # Jeśli modele zwracają 0 lub 1, możemy przyjąć,
    # że 1 = "Yes", a 0 = "No".
    # Obliczamy średnią ważoną:
    print(f"Predictions: pred1={pred1}, pred2={pred2}, pred3={pred3}, pred4={pred4}")
    print(f"Types: {type(pred1)}, {type(pred2)}, {type(pred3)}, {type(pred4)}")
    final_score = (
            pred1 * WEIGHTS["model_top_1"] +
            pred2 * WEIGHTS["model_top_2"] +
            pred3 * WEIGHTS["model_top_3"] +
            pred4 * WEIGHTS["model_top_4"]
    )

    # W tym miejscu "final_score" to liczba z przedziału [0, 1] (średnia ważona).
    # Przyjmijmy, że jeśli final_score >= 0.5 => "Yes", w przeciwnym razie "No".
    final_prediction = "Yes" if final_score >= 0.5 else "No"

    response = {
        "prediction": final_prediction
    }
    print("JESTEM2")
    return jsonify(response)


if __name__ == "__main__":
    # Uruchomienie serwera Flask (w trybie debug).
    # W środowisku produkcyjnym używaj np. Gunicorn + Nginx.
    app.run(debug=True)
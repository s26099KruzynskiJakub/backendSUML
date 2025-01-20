"""
Plik: main.py

Ten moduł prezentuje prosty serwer Flask przyjmujący dane (bez kolumn 'HadHeartAttack' i 'HadAngina'),
wywołujący trzy modele (LR, SVM, Ridge) i zwracający
ostateczną etykietę "Yes" lub "No" na podstawie średniej ważonej ich prognoz.

Modele (LR, SVM, Ridge) zostały wytrenowane przy użyciu PyCaret 3.3.2 i scikit-learn 1.4.2.
"""

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Wczytanie zapisanych modeli (pliki .pkl) - zaktualizowane nazwy modeli
print("JESTEM1")
model_lr = joblib.load("model_lr.pkl")  # LR
model_svm = joblib.load("model_svm.pkl")  # SVM
model_ridge = joblib.load("model_ridge.pkl")  # Ridge

# Wagi dla poszczególnych modeli (od najlepszego do najsłabszego) - zaktualizowane nazwy modeli i wagi
WEIGHTS = {
    "model_lr": 0.3334,  # LR
    "model_svm": 0.3333,  # SVM
    "model_ridge": 0.3333,  # Ridge
}

# Nazwy kolumn, które były wykorzystywane podczas trenowania (bez kolumn 'HadHeartAttack' i 'HadAngina')
COLUMNS = [
    "Sex",
    "GeneralHealth",
    "SleepHours",
    "RemovedTeeth",
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
    df_input = pd.DataFrame([data], columns=COLUMNS)

    # 1) Każdy z modeli zwraca etykietę "Yes" / "No" (lub 0/1).
    #    Tutaj wykonujemy predict() i mapujemy "Yes" -> 1, "No" -> 0:
    pred1 = 1 if model_lr.predict(df_input)[0] == "Yes" else 0
    pred2 = 1 if model_svm.predict(df_input)[0] == "Yes" else 0
    pred3 = 1 if model_ridge.predict(df_input)[0] == "Yes" else 0

    print(f"Predictions (0/1): {pred1}, {pred2}, {pred3}")

    # 2) Obliczamy średnią WAGOWANĄ
    final_score = (
        pred1 * WEIGHTS["model_lr"] +
        pred2 * WEIGHTS["model_svm"] +
        pred3 * WEIGHTS["model_ridge"]
    )

    # 3) Wyznaczamy finalną etykietę:
    final_prediction = "Yes" if final_score >= 0.5 else "No"

    # 4) *DODATKOWO* liczymy uśrednioną wartość predict_proba
    #    tylko dla tej klasy, którą finalnie przewidziano.
    if final_prediction == "Yes":
        prob1 = model_lr.predict_proba(df_input)[0][1]

        mean_prob = prob1

    else:  # final_prediction == "No"
        prob1 = model_lr.predict_proba(df_input)[0][0]

        mean_prob = prob1

    # 5) Konstruujemy odpowiedź.
    response = {
        "prediction": final_prediction,
        "weighted_score": round(final_score, 4),
        "mean_probability_of_that_class": round(mean_prob, 4)
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)

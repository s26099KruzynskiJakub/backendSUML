```markdown
# Aplikacja do predykcji zdarzeń sercowych

## Opis

Aplikacja backendowa, która na podstawie danych o pacjencie przewiduje, czy wystąpi u niego zdarzenie sercowe (zdefiniowane jako wystąpienie zawału serca - 'HadHeartAttack'). Aplikacja wykorzystuje trzy modele uczenia maszynowego:

*   **Regresja Logistyczna (LR)**
*   **Maszyna Wektorów Nośnych (SVM) z jądrem liniowym**
*   **Klasyfikator Ridge**

Modele zostały wytrenowane z użyciem biblioteki **PyCaret 3.3.2** oraz **scikit-learn 1.4.2** i zapisane do plików `.pkl`. Wyniki poszczególnych modeli są agregowane w celu uzyskania ostatecznej predykcji.

## Wymagania

*   Python 3.x
*   Flask
*   joblib
*   pandas
*   scikit-learn (zainstalowany razem z PyCaret)
*   PyCaret 3.3.2

## Instalacja

1.  **Sklonuj repozytorium:**

    ```bash
    git clone <adres_repozytorium>
    ```

    Lub pobierz pliki z repozytorium i rozpakuj je.
2.  **Przejdź do katalogu z aplikacją:**

    ```bash
    cd <nazwa_katalogu>
    ```

3.  **Zainstaluj wymagane biblioteki:**

    ```bash
    pip install -r requirements.txt
    ```

    **Uwaga:** Jeśli nie masz pliku `requirements.txt`, zainstaluj biblioteki ręcznie:

    ```bash
    pip install Flask joblib pandas scikit-learn pycaret==3.3.2
    ```

## Uruchomienie

1.  **Upewnij się, że pliki modeli (`model_lr.pkl`, `model_svm.pkl`, `model_ridge.pkl`) znajdują się w tym samym katalogu co plik `main.py`.**
2.  **Uruchom aplikację:**

    ```bash
    python main.py
    ```

    lub jeśli używasz `python3`:

    ```bash
    python3 main.py
    ```

3.  **Aplikacja zostanie uruchomiona i będzie dostępna pod adresem `http://127.0.0.1:5000/` (lub `http://localhost:5000/`).** W konsoli zobaczysz komunikat podobny do:

    ```
     * Serving Flask app 'main'
     * Debug mode: on
    WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
     * Running on [usunięto nieprawidłowy URL]
    Press CTRL+C to quit
     * Restarting with stat
    JESTEM1
     * Debugger is active!
     * Debugger PIN: ...-...-...
    ```

## Użycie

Aplikacja udostępnia jeden endpoint `/predict`, który przyjmuje żądania POST z danymi pacjenta w formacie JSON.

**Przykładowe żądanie:**

```
POST [usunięto nieprawidłowy URL]
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
```

**Przykładowa odpowiedź:**

```json
{
  "prediction": "No",
  "weighted_score": 0.3334,
  "mean_probability_of_that_class": 0.8765
}
```

**Opis odpowiedzi:**

*   `prediction`: Ostateczna predykcja ("Yes" lub "No").
*   `weighted_score`: Wartość ważonej średniej predykcji z trzech modeli.
*   `mean_probability_of_that_class`: Uśrednione prawdopodobieństwo przynależności do przewidzianej klasy.

## Uwagi

*   Aplikacja jest uruchomiona w trybie debugowania (`debug=True`). W środowisku produkcyjnym należy tę opcję wyłączyć.
*   **W środowisku produkcyjnym zamiast wbudowanego serwera Flask należy użyć serwera WSGI, takiego jak Gunicorn lub uWSGI.**
* Logi aplikacji informują o wczytywaniu modeli i wykonaniu predykcji (wyświetlając "JESTEM1" i "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").
```
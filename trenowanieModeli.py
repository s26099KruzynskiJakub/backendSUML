import pandas as pd
from pycaret.classification import *

# Wczytaj dane
df = pd.read_csv("dane.csv", sep=";")
df = df.drop(columns=["HadAngina"])

# Zidentyfikuj kolumny numeryczne i kategoryczne
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols.remove('HadHeartAttack')

# Inicjalizacja eksperymentu
exp = ClassificationExperiment()
exp.setup(
    data=df,
    target='HadHeartAttack',
    session_id=123,
    fold=10,
    use_gpu=False,
    categorical_features=categorical_cols,
    numeric_features=numeric_cols,
)
print("123")
# Porównaj modele
best_models = exp.compare_models(
    n_select=3,
    sort='Accuracy'
)
print("123")
# Wyświetl wyniki
results_df = exp.pull()
print("456")
print("----- Podsumowanie wszystkich porównanych modeli -----")
print(results_df)

# Finalizacja i zapis modeli
top_3 = best_models[:3]

for i, model in enumerate(top_3, start=1):
    final_m = exp.finalize_model(model)
    exp.save_model(final_m, f"model_{['lr', 'svm', 'ridge'][i-1]}")  # Zapisz jako model_lr, model_svm, model_ridge

print("\nZapisano 3 najlepsze modele jako: model_lr.pkl, model_svm.pkl, model_ridge.pkl")
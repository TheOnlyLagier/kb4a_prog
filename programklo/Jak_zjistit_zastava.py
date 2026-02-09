import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------- Načtení CSV a úprava dat ----------
X = []  # = vstupy
Y = []  # = výstupy

with open("programklo\heart.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Načtení vstupů
        inputs = [
            float(row["age"]),
            float(row["sex"]),
            float(row["cp"]),
            float(row["trestbps"]),
            float(row["chol"]),
            float(row["fbs"]),
            float(row["restecg"]),
            float(row["thalach"]),
            float(row["exang"]),
            float(row["oldpeak"]),
            float(row["slope"]),
            float(row["ca"]),
            float(row["thal"]),
        ]
        X.append(inputs)

        # Načtení výstupu
        Y.append(int(row["heart_disease"]))

# ---------- Ruční rozdělení na trénování a testování ----------
rows = len(X)
split = round(0.8 * rows)

trening_X = X[:split]
trening_Y = Y[:split]

test_X = X[split:]
test_Y = Y[split:]

# ---------- Neuronová síť ----------
neural_network = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation="relu",
    max_iter=2000,
    random_state=4
)

neural_network.fit(trening_X, trening_Y)

# ---------- Vyhodnocení ----------
predictions = neural_network.predict(test_X)

# Výpočet přesnosti
accuracy = accuracy_score(test_Y, predictions)
conf_matrix = confusion_matrix(test_Y, predictions)

print(f"Přesnost: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
# Neuronová síť predikující BMI kategorii 
# Jedná se pouze o učební ukázku - pro BMi je jinak využití neuronky nevhodné

import csv

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# ---------- Načtení CSV a úprava dat ----------
X = []  # = vstupy
Y = []  # = výstupy

with open("3. strojove_uceni/data/bmi.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        height = float(row["Height"])
        weight = float(row["Weight"])
        # Na vstupu mohou být jen číselné vstupy:
        if row["Gender"] == "Male":
            gender = 0
        else:
            gender = 1

        bmi_category = int(row["Index"])

        X.append([gender, height, weight])
        Y.append(bmi_category)


# ---------- Ruční rozdělení na trénování a testování ----------
rows = len(X)
split = round(0.8 * rows)

trening_X = X[:split]
trening_Y = Y[:split]

test_X = X[split:]
test_Y = Y[split:]

# ---------- Neuronová síť ----------
neural_network = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation="relu",
    max_iter=2000,
    random_state=4
)

neural_network.fit(trening_X, trening_Y)

# ---------- Vyhodnocení ----------
results = neural_network.predict(test_X)

correct = 0
for i in range(len(results)):
    if test_Y[i] == results[i]:
        correct += 1
print(correct / len(results))

print(confusion_matrix(test_Y, results))


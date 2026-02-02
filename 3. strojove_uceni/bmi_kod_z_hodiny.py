import csv
from sklearn.neural_network import MLPClassifier


X = [] # vstupy
Y = [] # výstupy


with open(r"3. strojove_uceni\data\bmi.csv", "r", encoding="utf-8") as file:
    for radek in csv.DictReader(file):
        vaha = int(radek["Weight"])
        vyska = int(radek["Height"])

        if radek["Gender"] == "Male":
            pohlavi = 0
        else:
            pohlavi = 1
   
        X.append([vyska, vaha, pohlavi])

        bmi_kategorie = int(radek["Index"])
        Y.append(bmi_kategorie)

delka = len(X)
X_trenovaci = X[:round(0.8*delka)]
X_test = X[round(0.8*delka):]
Y_trenovaci = Y[:round(0.8*delka)]
Y_test = Y[round(0.8*delka):]

neuronka = MLPClassifier(
    hidden_layer_sizes=(16,8),
    max_iter=2000,
    random_state=4
)

neuronka.fit(X_trenovaci, Y_trenovaci)

predikce = neuronka.predict(X_test)

print("Predikce:", predikce)
print("Správné odpovědi:", Y_test)
spravne = 0
for i in range(len(X_test)):
    if predikce[i] == Y_test[i]:
        spravne+=1
pocet = len(X_test)
print("Procento správných odpovědí:", round(spravne/pocet*100), "%")

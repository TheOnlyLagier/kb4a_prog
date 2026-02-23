import csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


X = []  #vstupy
Y = []  #výstupy

# Inicializace LabelEncoder pro textové sloupce
protocol_encoder = LabelEncoder()
encryption_encoder = LabelEncoder()
browser_encoder = LabelEncoder()


with open("programklo/cybersecurity_intrusion_data.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Extrakce textových sloupců pro LabelEncoder
protocol_types = [row["protocol_type"] for row in data]
encryption_types = [row["encryption_used"] for row in data]
browser_types = [row["browser_type"] for row in data]

# Fitting LabelEncoders
protocol_encoder.fit(protocol_types)
encryption_encoder.fit(encryption_types)
browser_encoder.fit(browser_types)


for row in data:
        
     # Převod textových hodnot na číselné pomocí LabelEncoder
    protocol = protocol_encoder.transform([row["protocol_type"]])[0]
    encryption = encryption_encoder.transform([row["encryption_used"]])[0]
    browser = browser_encoder.transform([row["browser_type"]])[0]

    #vstupy
    X.append([
        float(row["network_packet_size"]),
        protocol,
        int(row["login_attempts"]),
        float(row["session_duration"]),
        encryption,
        float(row["ip_reputation_score"]),
        int(row["failed_logins"]),
        browser,
        int(row["unusual_time_access"]),
        ])

    #vystupy
    Y.append(int(row["attack_detected"]))



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#neuronka
neural_network = MLPClassifier(
    hidden_layer_sizes=(64,32,16,8,4,2,1), #Nejvyssi pravdepodobnost ze 100% je pri 64,32,16,8,4,2,1
    activation="relu",
    max_iter=2000,
    random_state=42
)
neural_network.fit(X_train, y_train)

#FINALE
predictions = neural_network.predict(X_test)

#Presnost100% vzdy
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f"Přesnost: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the heart disease dataset
#Important!! download and load the dataset from Kaggle
data = pd.read_csv(r'C:\Users\ykurb\Downloads\heart.csv')

data
data.info()
data.describe()

# Create dummy variables for the categorical columns
data = pd.get_dummies(data, columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"], drop_first=True)

# Split the data into training and testing sets
X = data.drop(["target"], axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

data

# Apply KNN with k=3
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_train_scaled, y_train)
y_pred_knn1 = knn1.predict(X_test_scaled)

# Calculate the accuracy, precision, recall, and F1 score for KNN
knn1_accuracy = accuracy_score(y_test, y_pred_knn1)
knn1_precision = precision_score(y_test, y_pred_knn1)
knn1_recall = recall_score(y_test, y_pred_knn1)
knn1_f1 = f1_score(y_test, y_pred_knn1)

print("KNN1 accuracy:", knn1_accuracy)
print("KNN1 precision:", knn1_precision)
print("KNN1 recall:", knn1_recall)
print("KNN1 F1 score:", knn1_f1)

# Apply KNN with k=7
knn2 = KNeighborsClassifier(n_neighbors=7)
knn2.fit(X_train_scaled, y_train)
y_pred_knn2 = knn2.predict(X_test_scaled)

# Calculate the accuracy, precision, recall, and F1 score for KNN
knn2_accuracy = accuracy_score(y_test, y_pred_knn2)
knn2_precision = precision_score(y_test, y_pred_knn2)
knn2_recall = recall_score(y_test, y_pred_knn2)
knn2_f1 = f1_score(y_test, y_pred_knn2)

print("KNN2 accuracy:", knn2_accuracy)
print("KNN2 precision:", knn2_precision)
print("KNN2 recall:", knn2_recall)
print("KNN2 F1 score:", knn2_f1)

# Apply KNN with k=11
knn3 = KNeighborsClassifier(n_neighbors=11)
knn3.fit(X_train_scaled, y_train)
y_pred_knn3 = knn3.predict(X_test_scaled)

# Calculate the accuracy, precision, recall, and F1 score for KNN
knn3_accuracy = accuracy_score(y_test, y_pred_knn3)
knn3_precision = precision_score(y_test, y_pred_knn3)
knn3_recall = recall_score(y_test, y_pred_knn3)
knn3_f1 = f1_score(y_test, y_pred_knn3)

print("KNN3 accuracy:", knn3_accuracy)
print("KNN3 precision:", knn3_precision)
print("KNN3 recall:", knn3_recall)
print("KNN3 F1 score:", knn3_f1)

# Apply MLP with 1 hidden layer
mlp1 = MLPClassifier(hidden_layer_sizes=(32,))
mlp1.fit(X_train_scaled, y_train)
y_pred_mlp1 = mlp1.predict(X_test_scaled)

# Calculate the accuracy, precision, recall, and F1 score for MLP1
mlp1_accuracy = accuracy_score(y_test, y_pred_mlp1)
mlp1_precision = precision_score(y_test, y_pred_mlp1)
mlp1_recall = recall_score(y_test, y_pred_mlp1)
mlp1_f1 = f1_score(y_test, y_pred_mlp1)
print("MLP1 accuracy:", mlp1_accuracy)
print("MLP1 precision:", mlp1_precision)
print("MLP1 recall:", mlp1_recall)
print("MLP1 F1 score:", mlp1_f1)

# Apply MLP with 2 hidden layers
mlp2 = MLPClassifier(hidden_layer_sizes=(32, 32))
mlp2.fit(X_train_scaled, y_train)
y_pred_mlp2 = mlp2.predict(X_test_scaled)

# Calculate the accuracy, precision, recall, and F1 score for MLP2
mlp2_accuracy = accuracy_score(y_test, y_pred_mlp2)
mlp2_precision = precision_score(y_test, y_pred_mlp2)
mlp2_recall = recall_score(y_test, y_pred_mlp2)
mlp2_f1 = f1_score(y_test, y_pred_mlp2)
print("MLP2 accuracy:", mlp2_accuracy)
print("MLP2 precision:", mlp2_precision)
print("MLP2 recall:", mlp2_recall)
print("MLP2 F1 score:", mlp2_f1)

# Apply MLP with 3 hidden layers
mlp3 = MLPClassifier(hidden_layer_sizes=(32, 32, 32))
mlp3.fit(X_train_scaled, y_train)
y_pred_mlp3 = mlp3.predict(X_test_scaled)

# Calculate the accuracy, precision, recall, and F1 score for MLP3
mlp3_accuracy = accuracy_score(y_test, y_pred_mlp3)
mlp3_precision = precision_score(y_test, y_pred_mlp3)
mlp3_recall = recall_score(y_test, y_pred_mlp3)
mlp3_f1 = f1_score(y_test, y_pred_mlp3)

print("MLP3 accuracy:", mlp3_accuracy)
print("MLP3 precision:", mlp3_precision)
print("MLP3 recall:", mlp3_recall)
print("MLP3 F1 score:", mlp3_f1)

# Apply NB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Calculate the accuracy, precision, recall, and F1 score for NB
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)

print("NB accuracy:", nb_accuracy)
print("NB precision:", nb_precision)
print("NB recall:", nb_recall)
print("NB: F1 score", nb_f1)

# Compare the results
results = {"Accuracy": [knn1_accuracy,knn2_accuracy,knn3_accuracy, mlp1_accuracy, mlp2_accuracy,mlp3_accuracy,nb_accuracy],
           "Precision": [knn1_precision,knn2_precision,knn3_precision, mlp1_precision,mlp2_precision,mlp3_precision, nb_precision],
           "Recall": [knn1_recall,knn2_recall,knn3_recall, mlp1_recall,mlp2_recall,mlp3_recall, nb_recall],
           "F1 Score": [knn1_f1,knn2_f1,knn3_f1, mlp1_f1,mlp2_f1,mlp3_f1, nb_f1]}

df= pd.DataFrame(results, index=["KNN1","KNN2","KNN3", "MLP1","MLP2","MLP3", "NB"])
print(df)

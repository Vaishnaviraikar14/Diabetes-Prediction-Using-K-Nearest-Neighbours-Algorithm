import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def MarvellousWinePredictor(data_path):
    # Step 1: Get Data
    # Load data from WinePredictor.csv file into the python application
    data = pd.read_csv(data_path)
    print("Size of Actual dataset:", len(data))

    # Step 2: Clean, Prepare, and Manipulate data
    feature_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    print("Name of Features:", feature_names)

    X = data.drop("Class", axis=1)  # Features
    y = data["Class"]  # Target variable

    # Step 3: Train Data
    # Split data into 70% training and 30% testing
    data_train, data_test, target_train, target_test = train_test_split(X, y, test_size=0.3)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(data_train, target_train)

    # Step 4: Test Data
    new_data = [[13.5, 2.7, 2.68, 17.3, 120, 2.6, 2.9, 0.2, 1.1, 7.5, 0.65, 1.1, 575]]

    predictions = model.predict(new_data)
    print(f"Predicted class: {predictions[0]}")

    # Step 5: Calculate Accuracy
    def check_accuracy(data_train, target_train, data_test, target_test, k_values):
        accuracies = []

        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(data_train, target_train)
            predictions = model.predict(data_test)
            accuracy = accuracy_score(target_test, predictions)
            accuracies.append((k, accuracy))

        return accuracies

    # Example 
    k_values = [1, 3, 5, 7, 9]
    accuracies = check_accuracy(data_train, target_train, data_test, target_test, k_values)

    for k, accuracy in accuracies:
        print(f"Accuracy for k={k}: {accuracy}")

def main():
    print("------Marvellous Infosystem------")
    print("Machine learning Application")
    print("Wine predictor application using K nearest neighbor algorithm")
    MarvellousWinePredictor("WinePredictor.csv")

if __name__ == "__main__":
    main()

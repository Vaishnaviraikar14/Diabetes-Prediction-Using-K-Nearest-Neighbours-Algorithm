import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):

    #step 1 : load data
    data = pd.read_csv(data_path, index_col=0)
    print("Size pf Actual dataset",len(data))

    #step 2 : Clean, prepare and manipulate data
    feature_names = ['Whether','Temperature']
    print("Names of Features",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play 

    #creating labelencoder
    le = preprocessing.LabelEncoder()

    #Converting string labels into numbers.
    weather_encoded = le.fit_transform(whether)
    print(weather_encoded)

    #Converting string labels into numbers.
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)
    print(temp_encoded)

    #combining weather and temp into single listof tuples
    features = list(zip(weather_encoded,temp_encoded))

    #step3 : Train data
    model = KNeighborsClassifier(n_neighbors=3)
    
    #train the model using the training sets
    model.fit(features,label)

    #step4 : Test data
    predicted = model.predict([[0,2]])  # 0:Overcast, 2:Mild
    print(predicted)

def main():
    print("------Marvellous Infosystem------")
    print("Machine learning Application")

    print("Play predictor application using K nearest Knighbor algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()
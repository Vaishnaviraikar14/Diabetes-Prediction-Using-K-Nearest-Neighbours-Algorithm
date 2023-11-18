import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def AdvertisementPredictor():
    #load data
    data = pd.read_csv('Advertising.csv')
    print("Size of data set",data.shape)

    X = data[['TV','radio','newspaper']].values
    Y = data['sales'].values

    #train data
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

    model = LinearRegression()
    model.fit(X_train,Y_train)

    y_pred = model.predict(X_test)

    #findout goodness og fir ie. R squares
    mean_y = np.mean(Y_test)
    ss_tot = np.sum((Y_test - mean_y) ** 2)
    ss_res = np.sum((Y_test - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("R-squared value:", r2)

def main():
    print("-----Marvellous Infosystem-------")
    print("Supervisied Machine Learning")
    print("Linear Regression on Advertising data set")

    AdvertisementPredictor()

if __name__ == "__main__":
    main()


#output:
#PS C:\Users\VAISHNAVI\Desktop\Python_Assign16> python Assignment16.py
#-----Marvellous Infosystem-------
#Supervisied Machine Learning
#Linear Regression on Advertising data set
#Size of data set (200, 5)
#R-squared value: 0.8888269130179492
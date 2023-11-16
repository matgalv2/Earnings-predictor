import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

from utils import score


def randomForest():
    data_ML = read_csv("./resources/employments_ML2.csv", delimiter="\t")
    Y = data_ML["rate_per_hour"]
    X = data_ML.drop(columns="salary_monthly").drop(columns="rate_per_hour")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # randomForest = RandomForestRegressor(random_state=42, verbose=1)
    randomForest = RandomForestRegressor(random_state=42, verbose=1)
    randomForest.fit(X_train, y_train)
    return randomForest


def mlp() -> Sequential:
    data_ML = read_csv("./resources/employments_ML2.csv", delimiter="\t")

    Y = data_ML["rate_per_hour"]
    X = data_ML.drop(columns="salary_monthly").drop(columns="rate_per_hour")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    input_features_number = len(data_ML.columns)
    mlp = Sequential()
    mlp.add(Flatten())
    mlp.add(Dense((input_features_number + 1) / 2, activation="sigmoid"))
    mlp.add(Dense((input_features_number + 1) / 2, activation="sigmoid"))
    mlp.add(Dense(1))

    mlp.compile(loss='mse', metrics=['accuracy'])
    mlp.fit(X_train, y_train, epochs=10)

    predicted = mlp.predict(X_test).flatten()
    mseV, _ = mlp.evaluate(X_test, y_test)
    scoreV = score(predicted, y_test.to_list())

    # return mlp, mseV, scoreV
    return mlp


# def predict(x, model: RandomForestRegressor):
#     return model.predict(x)

# print(mlp())

x = [0.2017204541328603, 0.18, 0.0, 0.24, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]

y = 3.5
modelMLP = mlp()
print("MLP:")
print(modelMLP.predict([x]))
modelRF = randomForest()
print("Random forest:")
print(modelRF.predict([x]))

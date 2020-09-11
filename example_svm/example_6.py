"""
Date: 2020. 09. 08.
Programmer: MH
Description: Code for predicting School performance in 3rd period
"""
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.svm import SVR
import numpy as np


class SchoolPerformancePredictor:
    def __init__(self):
        pass

    def load_dataset(self, path):
        """
        To load dataset from local
        :param path: string, location of file
        :return:
        """
        self.data = pd.read_csv(path)
        print(self.data.head())
        print(self.data.info())

    def preprocess_dataset(self):
        """
        To shuffle data and normalize values range between 0 to 1 after changing textual data to numerical data
        :return:
        """
        self.data = self.data.sample(frac=1)
        enc = OrdinalEncoder()
        self.data = enc.fit_transform(self.data)
        min_max_scaler = MinMaxScaler()
        result = min_max_scaler.fit_transform(self.data)
        self.data = pd.DataFrame(result)
        print(self.data.head())

    def split_dataset(self, test_rate=0.2):
        """
        To split dataset to trainingset and testset considering the rate of test set
        :param test_rate: double, the rate of test set
        :return:
        """
        X = self.data.iloc[:,[1, 2, 3, 4, 10, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23, 24, 28, 29 ]]   # To set features except DEATH_EVENT attribute
        y = self.data.iloc[:, 32]    # To set DEATH_EVENT attribute to label
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_rate, random_state=42)
        print("# of Instances in Training Set: ", len(self.train_X))
        print("# of Instances in Test Set: ", len(self.test_X))

    def train_model(self):
        """
        To train regression model using training set
        :return:
        """
        self.svr = SVR(kernel="linear", C=5)    # To define model
        self.svr.fit(self.train_X, self.train_y)    # To train model
        dump(self.svr, "./School_Performance_Predictor.joblib")     # To save trained model to local

    def predict(self, x):
        """
        To predict qualities of input data
        :param x: dataframe or dict-shape data, features
        :return: ndarray, predicted result
        """
        self.svr = load("./School_Performance_Predictor.joblib")
        y_pred = self.svr.predict(x)
        return y_pred

    def evaluate_model(self, y_true, y_pred):
        """
        To evaluate trained model
        :param y_true: dict, label data driven from original data
        :param y_pred: dict, label data driven from predicted result
        :return: dict, performance results
        """
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        return {"MAE": mae, "MSE": mse}

    def finetune_model(self, kernel=None, degree=None, C=None, coef0=None, epsilon=None):
        """
        To change hyperparameters and train model again
        :param kernel: string, kernel of SVC
        :param degree: int, degree of polynomial kernel
        :param C: double, regularization
        :param coef0: double,
        :param epsilon: double
        :return:
        """
        if kernel is None:
            kernel = "rbf"
        if degree is None:
            degree = 3
        if C is None:
            C = 1.0
        if coef0 is None:
            coef0 = 0.0
        if epsilon is None:
            epsilon = 0.1
        self.svc = SVR(kernel=kernel, degree=degree, C=C, coef0=coef0, epsilon=epsilon)
        self.svc.fit(self.train_X, self.train_y)
        dump(self.svc, "./School_Performance_Predictor.joblib")     # To save trained model to local


if __name__ == '__main__':
    rwp = SchoolPerformancePredictor()
    rwp.load_dataset("../dataset/student/student-mat.csv")
    rwp.preprocess_dataset()
    rwp.split_dataset()
    rwp.train_model()
    pred_y = rwp.predict(rwp.test_X)
    result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")

    rwp.finetune_model(kernel="linear", C=1.0, epsilon=0.01)
    pred_y = rwp.predict(rwp.test_X)
    result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")

    rwp.finetune_model(kernel="rbf", C=1.0, epsilon=0.1)
    pred_y = rwp.predict(rwp.test_X)
    result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")

    rwp.finetune_model(kernel="rbf", C=1.0, epsilon=0.001)
    pred_y = rwp.predict(rwp.test_X)
    result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")
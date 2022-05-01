"""
Date: 2022. 04. 30.
Programmer: MH
Description: Code for predicting Wine Quality
"""
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np


class RedWineQualityPredictor:
    def __init__(self):
        pass

    def load_dataset(self, path_red, path_white):
        """
        To load dataset from local
        :param path: string, location of file
        :return:
        """
        self.data_red = pd.read_csv(path_red)
        self.data_white = pd.read_csv(path_white)
        self.data_red = self.data_red.sample(frac=1)
        self.data_white = self.data_white.sample(frac=1)

    def preprocess_dataset(self):
        """
        To shuffle data
        :return:
        """
        # Task 1. Feature Selection
        # To select 0th~11th attributes to the features
        self.features_red = self.data_red.iloc[:,:11]
        self.label_red = pd.DataFrame(self.data_red.iloc[:,11])
        self.features_white = self.data_white.iloc[:,:11]
        self.label_white = pd.DataFrame(self.data_white.iloc[:,11])
        print(self.features_red.info())
        print(self.label_red.info())
        print(self.features_white.info())
        print(self.label_white.info())

        # Task 2. Feature Scaling
        # To scale the value range of the whole features to 0..1 using MinMaxScaler
        scaler = MinMaxScaler()
        self.features_red = scaler.fit_transform(self.features_red)
        self.label_red = scaler.fit_transform(self.label_red)
        self.features_white = scaler.fit_transform(self.features_white)
        self.label_white = scaler.fit_transform(self.label_white)

    def split_dataset(self, test_rate=0.2):
        """
        To split dataset to trainingset and testset considering the rate of test set
        :param test_rate: double, the rate of test set
        :return:
        """
        train_x_white, test_x_white, train_y_white, test_y_white = train_test_split(self.features_white, self.label_white, test_size=test_rate, random_state=42)
        train_x_red, test_x_red, train_y_red, test_y_red = train_test_split(self.features_red, self.label_red, test_size=test_rate, random_state=42)
        print(type(train_x_white), type(train_x_red))
        print(train_x_red.shape, train_x_white.shape)
        self.train_X = np.concatenate((train_x_red, train_x_white))
        self.train_y = np.concatenate((train_y_red, train_y_white))
        self.test_X = np.concatenate((test_x_red, test_x_white))
        self.test_y = np.concatenate((test_y_red, test_y_white))
        print("# of Instances in Training Set: ", self.train_X.shape, self.train_y.shape)
        print("# of Instances in Test Set: ", self.test_X.shape, self.test_y.shape)

    def train_model(self):
        """
        To train regression model using training set
        :return:
        """
        self.svr = SVR(kernel="linear", C=5)    # To define model
        self.svr.fit(self.train_X, self.train_y)    # To train model
        dump(self.svr, "./RedWine_Quality_Regressor.joblib")     # To save trained model to local

    def predict(self, x):
        """
        To predict qualities of input data
        :param x: dataframe or dict-shape data, features
        :return: ndarray, predicted result
        """
        self.svr = load("./RedWine_Quality_Regressor.joblib")
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
        dump(self.svc, "./RedWine_Quality_Regressor.joblib")     # To save trained model to local


if __name__ == '__main__':
    rwp = RedWineQualityPredictor()
    rwp.load_dataset("../dataset/Red Wine Quality Dataset/winequality-red.csv", "../dataset/Red Wine Quality Dataset/winequality-white.csv")
    rwp.preprocess_dataset()
    rwp.split_dataset()
    rwp.train_model()
    pred_y = rwp.predict(rwp.test_X)
    result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")
    # rwp.finetune_model(kernel="linear", C=1.0, epsilon=0.01)
    # pred_y = rwp.predict(rwp.test_X)
    # result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    # print("MAE: ", result["MAE"])
    # print("MSE: ", result["MSE"])
    # print("\n\n")
    #
    # rwp.finetune_model(kernel="rbf", C=1.0, epsilon=0.1)
    # pred_y = rwp.predict(rwp.test_X)
    # result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    # print("MAE: ", result["MAE"])
    # print("MSE: ", result["MSE"])
    # print("\n\n")
    #
    # rwp.finetune_model(kernel="rbf", C=1.0, epsilon=0.001)
    # pred_y = rwp.predict(rwp.test_X)
    # result = rwp.evaluate_model(y_true=rwp.test_y, y_pred=pred_y)
    # print("MAE: ", result["MAE"])
    # print("MSE: ", result["MSE"])
    # print("\n\n")
"""
Date: 2020. 09. 08.
Programmer: MH
Description: Code for classifying bank-note authentication
"""
import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np


class BankNoteAuthenticator:
    def __init__(self):
        self.data = None

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
        To shuffle data
        :return:
        """
        self.data = self.data.sample(frac=1)
        print(self.data.head())

    def split_dataset(self, test_rate=0.2):
        """
        To split dataset to trainingset and testset considering the rate of test set
        :param test_rate: double, the rate of test set
        :return:
        """
        X = self.data.iloc[:,0:4]   # To set features except class attribute
        y = self.data.iloc[:, 4]    # To set class attribute to label
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_rate, random_state=42)
        print("# of Instances in Training Set: ", len(self.train_X))
        print("# of Instances in Test Set: ", len(self.test_X))

    def train_model(self):
        """
        To train model using training set
        :return:
        """
        self.svc = SVC(kernel='linear', C=1.0)   # To define model
        self.svc.fit(self.train_X, self.train_y)    # To train model
        dump(self.svc, "./banknote_authenticator.joblib")     # To save trained model to local

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: int, predicted result (0, 1)
        """
        self.svc = load("./banknote_authenticator.joblib")  # To load saved model file
        result = self.svc.predict(X)    # To predict labels about input data
        return result

    def evaluate_model(self, y_true, y_pred):
        """
        To evaluate trained model
        :param y_true: dict, label data driven from original data
        :param y_pred: dict, label data driven from predicted result
        :return: dict, performance results
        """
        result_acc = accuracy_score(y_true, y_pred)     # To compute accuracy
        result_prc = precision_score(y_true, y_pred, average="binary", pos_label=0) # To compute precision
        result_rec = recall_score(y_true, y_pred, average="binary", pos_label=0)    # To compute recall

        return {"Accuracy": result_acc, "Precision":result_prc, "Recall": result_rec}

    def finetune_model(self, kernel=None, degree=None, C=None, coef0=None):
        """
        To change hyperparameters and train model again
        :param kernel: string, kernel of SVC
        :param degree: int, degree of polynomial kernel
        :param C: double, regularization
        :param coef0:double,
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
        self.svc = SVC(kernel=kernel, degree=degree, C=C, coef0=coef0)  # To set svc algorithm using new set of hyperparameters
        self.svc.fit(self.train_X, self.train_y)    # To train model again
        dump(self.svc, "./banknote_authenticator.joblib")     # To save trained model to local


if __name__ == '__main__':
    bna = BankNoteAuthenticator()
    bna.load_dataset("../dataset/bill authentication/bill authentication.csv")
    bna.preprocess_dataset()
    bna.split_dataset()
    bna.train_model()
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y.tolist()))
    print(y_pred)
    result = bna.evaluate_model(bna.test_y, y_pred)
    print("Accuracy : ", result["Accuracy"])
    print("Precision : ", result["Precision"])
    print("Recall : ", result["Recall"])
    print("\n\n")

    bna.finetune_model(kernel="linear", C=10)
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y.tolist()))
    print(y_pred)
    result = bna.evaluate_model(bna.test_y, y_pred)
    print("Accuracy : ", result["Accuracy"])
    print("Precision : ", result["Precision"])
    print("Recall : ", result["Recall"])
    print("\n\n")

    bna.finetune_model(kernel="poly", degree=3, C=5)
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y.tolist()))
    print(y_pred)
    result = bna.evaluate_model(bna.test_y, y_pred)
    print("Accuracy : ", result["Accuracy"])
    print("Precision : ", result["Precision"])
    print("Recall : ", result["Recall"])
    print("\n\n")

    bna.finetune_model(kernel="poly", degree=8, C=5)
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y.tolist()))
    print(y_pred)
    result = bna.evaluate_model(bna.test_y, y_pred)
    print("Accuracy : ", result["Accuracy"])
    print("Precision : ", result["Precision"])
    print("Recall : ", result["Recall"])
    print("\n\n")
"""
Date: 2020. 09. 07.
Programmer: MH
Description: Code for classifying breast cancer is malignant or benign applying the real-valued features that are computed for each cell nucleus
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from joblib import dump, load
import numpy as np


class BreastCancerClassifier:
    def __init__(self):
        self.dataset = None
        self.train_X, self.test_X, self.train_y, self.test_y = None, None, None, None

    def load_dataset(self, path):
        """
        To load dataset from local location
        :param path: string, the file location
        :return: None.
        """
        self.dataset = pd.read_csv(path)    # To load csv file from local
        self.dataset.head()     # To show first five data in csv file
        self.dataset.info()
        pd.set_option("display.max_columns", None)
        print(self.dataset)

    def preprocess(self):
        """
        To preprocess dataset
        :return:
        """
        self.dataset = self.dataset.iloc[:, 1:12]
        self.dataset.info()

    def split_dataset(self):
        """
        To split dataset to features and label
        :return: None.
        """
        y = self.dataset.iloc[:, 0]     # To set "diagnosis" to label
        X = self.dataset.iloc[:, 1:11]   # To set other attributes as features
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Number of Data in Training Set: ", len(self.train_X), "  Number of Data in Test Set: ", len(self.test_X))

    def train_model(self):
        """
        To train model using trainingSet
        :return:
        """
        self.svc = SVC(kernel='linear') # To define model
        self.svc.fit(self.train_X, self.train_y)    # To train model
        dump(self.svc, "./breast_cancer_classifier.joblib")     # To save trained model to local

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: int, predicted result (0, 1)
        """
        self.svc = load("./breast_cancer_classifier.joblib")
        result = self.svc.predict(X)
        return result

    def evaluate_model(self):
        """
        To evaluate model
        :return: dict, performance results
        """
        y_pred = self.predict(self.test_X)
        result = classification_report(self.test_y, y_pred)
        acc_result = accuracy_score(self.test_y, y_pred)
        prec_result = precision_score(self.test_y, y_pred, average="binary", pos_label="M")
        rec_result = recall_score(self.test_y, y_pred, average="binary", pos_label="M")
        return {"Accuracy": acc_result, "Classification Report": result, "Precision": prec_result, "Recall": rec_result}

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
        self.svc = SVC(kernel=kernel, degree=degree, C=C, coef0=coef0)
        self.svc.fit(self.train_X, self.train_y)
        dump(self.svc, "./breast_cancer_classifier.joblib")     # To save trained model to local


if __name__ == '__main__':
    bcc = BreastCancerClassifier()
    bcc.load_dataset("../dataset/Breast Cancer Wisconsin Dataset/data.csv")
    bcc.preprocess()
    bcc.split_dataset()
    bcc.train_model()
    print(bcc.predict(bcc.test_X))
    print(np.array(bcc.test_y.values.tolist()))
    ev_result = bcc.evaluate_model()
    print("Applied Hyperparameters: Kernel-> Linear")
    print(ev_result["Accuracy"])
    print(ev_result["Precision"])
    print(ev_result["Recall"])

    print("Applied Hyperparameters: Kernel -> Poly, Degree -> 8")
    bcc.finetune_model(kernel="poly", degree=8)
    ev_result = bcc.evaluate_model()
    print(ev_result["Accuracy"])
    print(ev_result["Precision"])
    print(ev_result["Recall"])

    print("Applied Hyperparameters: Kernel -> RBF")
    bcc.finetune_model(kernel="rbf")
    ev_result = bcc.evaluate_model()
    print(ev_result["Accuracy"])
    print(ev_result["Precision"])
    print(ev_result["Recall"])



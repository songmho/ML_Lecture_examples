"""
Date: 2020. 09. 07.
Programmer: MH
Description: Code for recognizing identification from image
"""
from joblib import dump, load
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class FaceRecognizer:
    def __init__(self):
        pass

    def load_data(self, min_faces=60):
        """
        To load data from sklearn dataset
        :param min_faces: int, the minimun number of faces per a person
        :return:
        """
        self.data = fetch_lfw_people(min_faces_per_person=min_faces)
        self.features = self.data.data
        self.images = self.data.images
        self.labels = self.data.target
        self.names = self.data.target_names

        print(self.names)
        print("# of Dataset: ", len(self.labels))

    def split_data(self, test_rate=0.2):
        """
        To split dataset to training set and test set
        :param test_rate: double, the percentage of test set
        :return:
        """
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.features, self.labels,
                                                                                test_size=test_rate, random_state=42)
        _, self.img_x,_, self.img_y = train_test_split(self.features, self.images,
                                                                                test_size=test_rate, random_state=42)
        print("# of Instances in Training Set: ", len(self.train_X))
        print("# of Instances in Test Set: ", len(self.test_X))

    def train_model(self):
        """
        To train model using training set
        :return:
        """
        self.svc = SVC(kernel='rbf', class_weight="balanced")   # To define model
        self.svc.fit(self.train_X, self.train_y)    # To train model
        dump(self.svc, "./face_recognizer.joblib")     # To save trained model to local

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: int, predicted result (0, 1)
        """
        self.svc = load("./face_recognizer.joblib")
        result = self.svc.predict(X)
        return result

    def evaluate_model(self, y_true, y_pred):
        """
        To evaluate trained model
        :param y_true: dict, label data driven from original data
        :param y_pred: dict, label data driven from predicted result
        :return: dict, performance results
        """
        result_acc = accuracy_score(y_true, y_pred)
        # result_prc = precision_score(y_true, y_pred)
        # result_rec = recall_score(y_true, y_pred)
        result = classification_report(y_pred=y_pred, y_true=y_true,
                              target_names=self.data.target_names)

        return {"Accuracy": result_acc, "Classification Report": result}

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
        dump(self.svc, "./face_recognizer.joblib")     # To save trained model to local


if __name__ == '__main__':
    fr = FaceRecognizer()
    fr.load_data()
    fr.split_data()
    print(fr.test_y)
    print("[", end="")
    for t in fr.test_y:
        print(fr.names[t], end=", ")
    print("]")
    fr.train_model()
    y_pred = fr.predict(fr.test_X)
    print(np.array(fr.test_y))
    print(y_pred)
    print("[", end="")
    for t in y_pred:
        print(fr.names[t], end=", ")
    print("]")
    result = fr.evaluate_model(fr.test_y, y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Classification Report: \n", result["Classification Report"])
    print()
    #
    # fr.finetune_model(C=5)
    # y_pred = fr.predict(fr.test_X)
    # print(np.array(fr.test_y))
    # print(y_pred)
    # result = fr.evaluate_model(fr.test_y, y_pred)
    # print("Accuracy: ", result["Accuracy"])
    # print("Classification Report: \n", result["Classification Report"])
    # print()
    #
    # fr.finetune_model(kernel="linear", C=1)
    # y_pred = fr.predict(fr.test_X)
    # print(np.array(fr.test_y))
    # print(y_pred)
    # result = fr.evaluate_model(fr.test_y, y_pred)
    # print("Accuracy: ", result["Accuracy"])
    # print("Classification Report: \n", result["Classification Report"])
    # print()
    #
    # fr.finetune_model(kernel="linear", C=5)
    # y_pred = fr.predict(fr.test_X)
    # print(np.array(fr.test_y))
    # print(y_pred)
    # result = fr.evaluate_model(fr.test_y, y_pred)
    # print("Accuracy: ", result["Accuracy"])
    # print("Classification Report: \n", result["Classification Report"])
    # print()
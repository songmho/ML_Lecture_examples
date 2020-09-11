"""
Date: 2020. 09. 07.
Programmer: MH
Description: Code for regressing boston housing dataset
"""
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import numpy as np
from sklearn import tree


class DiabetesPatientClassifier:
    def __init__(self):
        self.data = None
        self.feature_names = None
        self.class_names = None

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
        X = self.data.iloc[:, 0:8]   # To set features except outcome attribute
        y = self.data.iloc[:, 8]    # To set outcome attribute to label
        self.feature_names = X.columns
        self.class_names = y.unique().astype('str')
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_rate, random_state=42)
        print("# of Instances in Training Set: ", len(self.train_X))
        print("# of Instances in Test Set: ", len(self.test_X))

    def train_model(self):
        """
        To train model using training set
        :return:
        """
        self.dtc = DecisionTreeClassifier(max_depth=10)   # To define model
        self.dtc.fit(self.train_X, self.train_y)    # To train model
        print(self.dtc.get_n_leaves(), self.dtc.get_depth())
        dump(self.dtc, "./diabetes_patient_classifier.joblib")     # To save trained model to local

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: int, predicted result (0, 1)
        """
        self.dtc = load("./diabetes_patient_classifier.joblib")
        result = self.dtc.predict(X)
        return result

    def evaluate_model(self, y_true, y_pred):
        """
        To evaluate trained model
        :param y_true: dict, label data driven from original data
        :param y_pred: dict, label data driven from predicted result
        :return: dict, performance results
        """
        result_acc = accuracy_score(y_true, y_pred)
        result_prc = precision_score(y_true, y_pred, average="binary", pos_label=1)
        result_rec = recall_score(y_true, y_pred, average="binary", pos_label=1)

        return {"Accuracy": result_acc, "Precision": result_prc, "Recall": result_rec}

    def finetune_model(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_leaf_nodes=None ):
        """
        To change hyperparameters and train model again
        :param kernel: string, kernel of SVC
        :param degree: int, degree of polynomial kernel
        :param C: double, regularization
        :param coef0:double,
        :return:
        """
        self.dtc = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
        self.dtc.fit(self.train_X, self.train_y)
        dump(self.dtc, "./diabetes_patient_classifier.joblib")     # To save trained model to local

    def visualize_tree(self):
        """
        To save *.dot file of trained model
        :return:
        """
        tree.export_graphviz(self.dtc, out_file="diabetes_patient_classifier.dot", feature_names=self.feature_names,
                             class_names=self.class_names, filled=True)


if __name__ == '__main__':
    bcc = DiabetesPatientClassifier()
    bcc.load_dataset("../dataset/Pima Indians Diabetes Dataset/diabetes.csv")
    bcc.preprocess_dataset()
    bcc.split_dataset()
    bcc.train_model()
    bcc.visualize_tree()
    y_pred = bcc.predict(bcc.test_X)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print(np.array(bcc.test_y))
    print(y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bcc.finetune_model(max_depth=30)
    y_pred = bcc.predict(bcc.test_X)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print(np.array(bcc.test_y))
    print(y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bcc.finetune_model(max_depth=10, criterion="entropy")
    y_pred = bcc.predict(bcc.test_X)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print(np.array(bcc.test_y))
    print(y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bcc.finetune_model(max_depth=30, criterion="entropy")
    y_pred = bcc.predict(bcc.test_X)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print(np.array(bcc.test_y))
    print(y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

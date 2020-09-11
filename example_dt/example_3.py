"""
Date: 2020. 09. 08.
Programmer: MH
Description: Code for classifying bank-note authentication
"""
import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import tree


class BankNoteAuthenticator:
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
        X = self.data.iloc[:,0:4]   # To set features except class attribute
        y = self.data.iloc[:, 4]    # To set class attribute to label
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
        dump(self.dtc, "./banknote_authenticator.joblib")     # To save trained model to local

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: int, predicted result (0, 1)
        """
        self.dtc = load("./banknote_authenticator.joblib")  # To load saved model file
        result = self.dtc.predict(X)    # To predict labels about input data
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
        dump(self.dtc, "./banknote_authenticator.joblib")     # To save trained model to local

    def visualize_tree(self):
        """
        To save *.dot file of trained model
        :return:
        """
        tree.export_graphviz(self.dtc, out_file="banknote_authenticator.dot", feature_names=self.feature_names,
                             class_names=self.class_names, filled=True)



if __name__ == '__main__':
    bna = BankNoteAuthenticator()
    bna.load_dataset("../dataset/bill authentication/bill authentication_4.csv")
    bna.preprocess_dataset()
    bna.split_dataset()
    bna.train_model()
    bna.visualize_tree()
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y))
    print(y_pred)
    result = bna.evaluate_model(y_true=bna.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bna.finetune_model(max_depth=30)
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y))
    print(y_pred)
    result = bna.evaluate_model(y_true=bna.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bna.finetune_model(max_depth=10, criterion="entropy")
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y))
    print(y_pred)
    result = bna.evaluate_model(y_true=bna.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bna.finetune_model(max_depth=30, criterion="entropy")
    y_pred = bna.predict(bna.test_X)
    print(np.array(bna.test_y))
    print(y_pred)
    result = bna.evaluate_model(y_true=bna.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

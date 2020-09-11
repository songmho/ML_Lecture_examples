"""
Date: 2020. 09. 07.
Programmer: MH
Description: Code for regressing boston housing dataset
"""
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from sklearn import tree


class TitanicSurvivorClassifier:
    def __init__(self):
        self.data = None
        self.feature_names = None
        self.class_names = None

    def load_dataset(self,train_path, test_path, test_label_path):
        """
        To load dataset from local
        :param path: string, location of file
        :return:
        """
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        test_labels = pd.read_csv(test_label_path).iloc[:, 1]
        self.test_data = pd.concat([test_labels, self.test_data], axis=1)
        print(self.train_data.head(10))

    def preprocess_dataset(self):
        """
        To shuffle data
        :return:
        """
        self.train_data = self.train_data.dropna()
        self.test_data = self.test_data.dropna()
        self.train_X = self.train_data.iloc[:, [2, 4, 5, 6, 7, 9]]   # To set features for training set
        self.train_y = self.train_data.iloc[:, 1]   # To set label for training set
        self.test_X = self.test_data.iloc[:, [2, 4, 5, 6, 7, 9]]   # To set features for test set
        self.test_y = self.test_data.iloc[:, 0]   # To set label for test set
        self.feature_names = self.train_X.columns
        self.class_names = self.train_y.unique().astype('str')
        enc = LabelEncoder()
        self.train_X = self.train_X.copy()  # To prevent warning message
        self.test_X = self.test_X.copy()  # To prevent warning message
        self.train_X.loc[:, 'Sex'] = enc.fit_transform(self.train_X['Sex']) # To transform "male", "female" to 0, 1
        self.test_X.loc[:, "Sex"] = enc.fit_transform(self.test_X["Sex"])   # To transform "male", "female" to 0, 1

        # To change type of float type data to float32
        self.train_X.loc[:, "Age"] = self.train_X["Age"].astype("float32")
        self.train_X.loc[:, "Fare"] = self.train_X["Fare"].astype("float32")
        self.test_X.loc[:, "Age"] = self.test_X["Age"].astype("float32")
        self.test_X.loc[:, "Fare"] = self.test_X["Fare"].astype("float32")
        print(self.train_X.head(10))

    def train_model(self):
        """
        To train model using training set
        :return:
        """
        self.dtc = DecisionTreeClassifier(max_depth=10)   # To define model
        self.dtc.fit(self.train_X, self.train_y)    # To train model
        print(self.dtc.get_n_leaves(), self.dtc.get_depth())
        dump(self.dtc, "./Titanic_Survivor_Classifier.joblib")     # To save trained model to local

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: int, predicted result (0, 1)
        """
        self.dtc = load("./Titanic_Survivor_Classifier.joblib")
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
        dump(self.dtc, "./Titanic_Survivor_Classifier.joblib")     # To save trained model to local

    def visualize_tree(self):
        """
        To save *.dot file of trained model
        :return:
        """
        tree.export_graphviz(self.dtc, out_file="Titanic_Survivor_Classifier.dot", feature_names=self.feature_names,
                             class_names=self.class_names, filled=True)


if __name__ == '__main__':
    bcc = TitanicSurvivorClassifier()
    bcc.load_dataset("../dataset/Titanic/train.csv", "../dataset/Titanic/test.csv", "../dataset/Titanic/gender_submission.csv")
    bcc.preprocess_dataset()
    bcc.train_model()
    bcc.visualize_tree()
    y_pred = bcc.predict(bcc.test_X)
    print(np.array(bcc.test_y))
    print(y_pred)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bcc.finetune_model(max_depth=30)
    y_pred = bcc.predict(bcc.test_X)
    print(np.array(bcc.test_y))
    print(y_pred)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bcc.finetune_model(max_depth=10, criterion="entropy")
    y_pred = bcc.predict(bcc.test_X)
    print(np.array(bcc.test_y))
    print(y_pred)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

    bcc.finetune_model(max_depth=30, criterion="entropy")
    y_pred = bcc.predict(bcc.test_X)
    print(np.array(bcc.test_y))
    print(y_pred)
    result = bcc.evaluate_model(y_true=bcc.test_y, y_pred=y_pred)
    print("Accuracy: ", result["Accuracy"])
    print("Precision: ", result["Precision"])
    print("Recall: ", result["Recall"])
    print("\n\n")

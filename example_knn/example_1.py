"""
Date: 2022. 01. 20.
Programmer: MH
Description: Code for classifying breast cancer using K Nearest Neighbor Classification
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from joblib import dump, load
import numpy as np
from sklearn.model_selection import GridSearchCV


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
        print(self.dataset.head())     # To show first five data in csv file
        self.dataset.info()
        pd.set_option("display.max_columns", None)
        # print(self.dataset)

    def preprocess(self):
        """
        To preprocess dataset
        :return:
        """
        self.dataset = self.dataset.iloc[:, 1:13]
        # self.dataset = self.dataset.sample(frac=1)
        print(self.dataset.head())
        self.dataset.info()

    def split_dataset(self, scale=True):
        """
        To split dataset to features and label
        :return: None.
        """
        y = self.dataset.iloc[:, 0]     # To set "diagnosis" to label
        X = self.dataset.iloc[:, 1:12]   # To set other attributes as features
        if scale:
            min_max_scaler = MinMaxScaler()
            X_result = min_max_scaler.fit_transform(X)
            X = pd.DataFrame(X_result, columns=X.columns)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Number of Data in Training Set: ", len(self.train_X), "  Number of Data in Test Set: ", len(self.test_X))

    def train_model(self, break_ties=False):
        """
        To train model using trainingSet
        :return:
        """
        self.knn = KNeighborsClassifier(n_neighbors=5, weights="uniform")  # To define model
        self.knn.fit(self.train_X, self.train_y)    # To train model
        dump(self.knn, "./breast_cancer_classifier.joblib")     # To save trained model to local

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: int, predicted result (0, 1)
        """
        self.knn = load("./breast_cancer_classifier.joblib")
        result = self.knn.predict(X)
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

    def finetune_model(self, n_neighbors=5, weights="uniform", algorithm='ball_tree'):
        """
        To fine-tune model changing hyperparameters
        :param n_neighbors:
        :param weights:
        :param algorithm:
        :return:
        """
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)  # To define model
        self.knn.fit(self.train_X, self.train_y)
        dump(self.knn, "./breast_cancer_classifier.joblib")     # To save trained model to local

    def grid_search(self, param_grid ):
        self.knn = KNeighborsClassifier()
        gs = GridSearchCV(estimator=self.knn, param_grid=param_grid, scoring="accuracy", cv=5)
        gs.fit(self.train_X, self.train_y)
        print(gs.cv_results_["params"])
        print(gs.cv_results_["mean_test_score"])
        print(gs.best_score_)
        print(gs.best_params_)


if __name__ == '__main__':
    bcc = BreastCancerClassifier()
    bcc.load_dataset("../dataset/Breast Cancer Wisconsin Dataset/data.csv")
    bcc.preprocess()
    bcc.split_dataset()
    bcc.train_model()
    # print(bcc.predict(bcc.test_X))
    # print(np.array(bcc.test_y.values.tolist()))
    ev_result = bcc.evaluate_model()
    print("Applied Hyperparameters: # of Neighbors: 5, Weights: Uniform, Algorithm: Auto")
    print(ev_result["Accuracy"])
    print(ev_result["Precision"])
    print(ev_result["Recall"])

    print("\n\n")

    bcc = BreastCancerClassifier()
    bcc.load_dataset("../dataset/Breast Cancer Wisconsin Dataset/data.csv")
    bcc.preprocess()
    bcc.split_dataset()
    bcc.finetune_model(n_neighbors=7, weights="uniform", algorithm='ball_tree')
    ev_result = bcc.evaluate_model()
    print("Applied Hyperparameters: # of Neighbors: 7, Weights: Distance, Algorithm: ball_tree")
    print(ev_result["Accuracy"])
    print(ev_result["Precision"])
    print(ev_result["Recall"])
    print("\n\n")

    bcc = BreastCancerClassifier()
    bcc.load_dataset("../dataset/Breast Cancer Wisconsin Dataset/data.csv")
    bcc.preprocess()
    bcc.split_dataset()
    bcc.finetune_model(n_neighbors=15, weights="uniform", algorithm='ball_tree')
    ev_result = bcc.evaluate_model()
    print("Applied Hyperparameters: # of Neighbors: 15, Weights: Distance, Algorithm: ball_tree")
    print(ev_result["Accuracy"])
    print(ev_result["Precision"])
    print(ev_result["Recall"])
    print("\n\n")



    bcc = BreastCancerClassifier()
    bcc.load_dataset("../dataset/Breast Cancer Wisconsin Dataset/data.csv")
    bcc.preprocess()
    bcc.split_dataset()
    bcc.grid_search(param_grid={'n_neighbors': [3, 5, 7, 10, 15],
                                'weights': ["uniform", "distance"],
                                "algorithm": ["ball_tree", "kd_tree", "brute"]})
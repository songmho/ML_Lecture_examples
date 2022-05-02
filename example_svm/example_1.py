"""
Date: 2022. 05. 01.
Programmer: MH
Description: Code for classifying breast cancer is malignant or benign applying the real-valued features that are computed for each cell nucleus
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from joblib import dump, load
import numpy as np
from sklearn.model_selection import GridSearchCV


class BreastCancerClassifier:
    def __init__(self):
        self.dataset = None
        self.train_X, self.test_X, self.train_y, self.test_y = None, None, None, None
        self.th_corr = 0.5
        self.test_rate = 0.2

    def load_dataset(self, path):
        """
        To load dataset from local location
        :param path: string, the file location
        :return: None.
        """
        self.dataset = pd.read_csv(path)    # To load csv file from local
        pd.set_option("display.max_columns", None)

    def preprocess(self):
        """
        To preprocess dataset
        :return:
        """
        # Task 1. Encode String Data to Numerical Data
        ord_enc = OrdinalEncoder()
        result = ord_enc.fit_transform(np.array(self.dataset.iloc[:, 1]).reshape(-1, 1))
        self.dataset.iloc[:, 1] = pd.Series(result.reshape(1, -1).flatten())

        # Task 2. Select Features
        correlations = self.dataset.corr().loc[:, "diagnosis"]
        selected_attrs = correlations[(correlations >= self.th_corr) | (correlations<=-self.th_corr)].keys()
        self.dataset = self.dataset.loc[:, selected_attrs]

        self.features = self.dataset.iloc[:, 1:]
        self.label = self.dataset.iloc[:, 0]

        # Task 3. Feature Scaling
        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(self.features)

    def split_dataset(self):
        """
        To split dataset to features and label
        :return: None.
        """
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.features, self.label, test_size=self.test_rate, random_state=42)
        pd_train = pd.merge(pd.DataFrame(self.train_X), self.train_y,right_index = True,left_index = True)
        pd_test = pd.merge(pd.DataFrame(self.test_X), self.test_y,right_index = True,left_index = True)
        pd_train.to_csv(r"E:\2. Project\Python\ML_Lecture_examples\dataset\Breast Cancer Wisconsin Dataset\train.csv")
        pd_test.to_csv(r"E:\2. Project\Python\ML_Lecture_examples\dataset\Breast Cancer Wisconsin Dataset\test.csv")

        print(self.train_X.shape)
        print(self.test_X.shape)
        print(self.train_y.shape)
        print(self.test_y.shape)

    def train_model(self):
        """
        To train model using trainingSet
        :return:
        """
        self.svc = SVC(kernel='poly', gamma="scale", degree=3, coef0=0)  # To define model
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
        print(self.test_y)
        print(y_pred)
        result = classification_report(self.test_y, y_pred)
        acc_result = accuracy_score(self.test_y, y_pred)
        prec_result = precision_score(self.test_y, y_pred, average="binary", pos_label=1)
        rec_result = recall_score(self.test_y, y_pred, average="binary", pos_label=1)
        f1_result = f1_score(self.test_y, y_pred)
        return {"Accuracy": acc_result, "Classification Report": result,
                "Precision": prec_result, "Recall": rec_result, "f1_score": f1_result}


if __name__ == '__main__':
    bcc = BreastCancerClassifier()
    bcc.load_dataset("../dataset/Breast Cancer Wisconsin Dataset/data.csv")
    bcc.preprocess()
    bcc.split_dataset()
    bcc.train_model()
    ev_result = bcc.evaluate_model()
    print("Accuracy: ", ev_result["Accuracy"])
    print("Precision: ", ev_result["Precision"])
    print("Recall: ", ev_result["Recall"])
    print("F1 Score: ", ev_result["f1_score"])
"""
Date: 2020. 09. 08.
Programmer: MH
Description: Code for predicting ramen rating
"""
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn import tree


class RamenRatingPredictor:
    def __init__(self):
        self.feature_names = None
        self.class_names = None

    def load_dataset(self, path):
        """
        To load dataset from local
        :param path: String, location of file
        :return:
        """
        self.data = pd.read_csv(path)
        print(self.data.head(10))

    def preprocess_dataset(self):
        """
        To select features and shuffle data
        :return:
        """
        self.data = self.data.iloc[:,[1, 3, 4, 5]]
        self.feature_names = self.data.iloc[:, :3].columns
        self.class_names = self.data.iloc[:, 3].unique().astype('str')
        self.data = self.data.dropna()
        self.data = self.data.sample(frac=1)
        print(self.data.info())
        ode = OrdinalEncoder()
        data = ode.fit_transform(self.data)
        self.data = pd.DataFrame(data, columns=self.data.columns)
        print(self.data.info())
        print(self.data.head(10))

    def split_dataset(self, test_rate=0.2):
        """
        To split dataset to trainingset and testset considering the rate of test set
        :param test_rate: double, the rate of test set
        :return:
        """
        X = self.data.iloc[:, :3]
        y = self.data.iloc[:, 3]

        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, test_size=test_rate, random_state=42)

        print("# of Instances in Training Set: ", len(self.train_X))
        print("# of Instances in Test Set: ", len(self.test_X))

    def train_model(self):
        """
        To train model using training set
        :return:
        """
        self.dtr = DecisionTreeRegressor(max_depth=30)
        self.dtr.fit(self.train_X, self.train_y)
        print(self.dtr.get_n_leaves(), self.dtr.get_depth())
        dump(self.dtr, "./Ramen_Rating_Predictor.joblib")

    def predict(self, X):
        """
        To predict diagnosis result derived X
        :param X: dataframe, dict, an instance for predicting label
        :return: Array, predicted result
        """
        self.dtr = load("./Ramen_Rating_Predictor.joblib")
        result = self.dtr.predict(X)
        return result

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

    def finetune_model(self, criterion="mse", max_depth=None, min_samples_leaf=1, max_leaf_nodes=None ):
        """
        To change hyperparameters and train model again
        :param kernel: string, kernel of SVC
        :param degree: int, degree of polynomial kernel
        :param C: double, regularization
        :param coef0:double,
        :return:
        """
        self.dtr = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
        self.dtr.fit(self.train_X, self.train_y)

        print(self.dtr.get_n_leaves(), self.dtr.get_depth())
        dump(self.dtr, "./Ramen_Rating_Predictor.joblib")     # To save trained model to local

    def visualize_tree(self):
        """
        To save *.dot file of trained model
        :return:
        """
        tree.export_graphviz(self.dtr, out_file="Ramen_Rating_Predictor.dot", feature_names=self.feature_names,
                             class_names=self.class_names, filled=True)



if __name__ == '__main__':
    rrp = RamenRatingPredictor()
    rrp.load_dataset("../dataset/Ramen Ratings/ramen-ratings.csv")
    rrp.preprocess_dataset()
    rrp.split_dataset()
    rrp.train_model()
    rrp.visualize_tree()
    pred_y = rrp.predict(rrp.test_X)
    print(np.array(rrp.test_y))
    print(pred_y)
    result = rrp.evaluate_model(y_true=rrp.test_y, y_pred=pred_y)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")

    rrp.finetune_model(max_depth=None)
    y_pred = rrp.predict(rrp.test_X)
    print(np.array(rrp.test_y))
    print(y_pred)
    result = rrp.evaluate_model(y_true=rrp.test_y, y_pred=y_pred)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")

    rrp.finetune_model(max_depth=30, criterion="friedman_mse")
    y_pred = rrp.predict(rrp.test_X)
    print(np.array(rrp.test_y))
    print(y_pred)
    result = rrp.evaluate_model(y_true=rrp.test_y, y_pred=y_pred)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")


    rrp.finetune_model(max_depth=100, criterion="friedman_mse")
    y_pred = rrp.predict(rrp.test_X)
    print(np.array(rrp.test_y))
    print(y_pred)
    result = rrp.evaluate_model(y_true=rrp.test_y, y_pred=y_pred)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")



    rrp.finetune_model(max_depth=30, criterion="mae")
    y_pred = rrp.predict(rrp.test_X)
    print(np.array(rrp.test_y))
    print(y_pred)
    result = rrp.evaluate_model(y_true=rrp.test_y, y_pred=y_pred)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")


    rrp.finetune_model(max_depth=100, criterion="mae")
    y_pred = rrp.predict(rrp.test_X)
    print(np.array(rrp.test_y))
    print(y_pred)
    result = rrp.evaluate_model(y_true=rrp.test_y, y_pred=y_pred)
    print("MAE: ", result["MAE"])
    print("MSE: ", result["MSE"])
    print("\n\n")

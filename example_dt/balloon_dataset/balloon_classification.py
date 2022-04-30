"""
Class for classifying balloon inflation based on Decision Tree Classification
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


class BalloonInflationClassification:
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

    def preprocess_dataset(self):
        """
        To preprocess dataset
        :return:
        """
        print(self.data.shape)
        for idx in range(self.data.shape[-1]):
            self.data.iloc[:, idx], _ = pd.factorize(self.data.iloc[:, idx])  # To factorize attributes

    def split_dataset(self):
        """
        To split dataset to trainingset and testset considering the rate of test set
        :param test_rate: double, the rate of test set
        :return:
        """
        self.train_X = self.data.iloc[:, :4]   # To set features except inflated attribute
        self.train_y = self.data.iloc[:, 4]    # To set inflated attribute to label
        self.feature_names = self.train_X.columns
        self.class_names = self.train_y.unique().astype('str')

    def train_model(self):
        """
        To train model using training set
        :return:
        """
        self.dtc = DecisionTreeClassifier(max_depth=10)   # To define model
        self.dtc.fit(self.train_X, self.train_y)    # To train model
        print(self.dtc.get_n_leaves(), self.dtc.get_depth())

    def visualize_tree(self):
        """
        To save *.dot file of trained model
        :return:
        """
        tree.export_graphviz(self.dtc, out_file="balloon_decision_tree.dot", feature_names=self.feature_names,
                             class_names=self.class_names, filled=True) # To save tree to *.dot file
        # To transform *.dot file to *.png file using graphviz web site
        # https://dreampuf.github.io/GraphvizOnline/


if __name__ == '__main__':
    bcc = BalloonInflationClassification()
    bcc.load_dataset("./Dataset/yellow-small+adult-stretch.csv")
    bcc.preprocess_dataset()
    bcc.split_dataset()
    bcc.train_model()
    bcc.visualize_tree()

"""
Date: 2020. 09. 09.
Programmer: MH
Description: Code for making customer groups considering purchase items
"""
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import numpy as np

pd.set_option('display.max_columns', 500)


class WholesaleCustomerCluster:
    def __init__(self):
        self.data = None

    def load_dataset(self, path):
        """
        To load dataset from local
        :param path: string, dataset path
        :return:
        """
        self.data = pd.read_csv(path)   # To load data
        print(self.data.head())
        print(self.data.info())

    def preprocess_dataset(self):
        """
        To normalize whole data from 0 to 1
        :return:
        """
        mms = MinMaxScaler()
        d = mms.fit_transform(self.data)    # To normalize data from 0 to 1
        self.data = pd.DataFrame(data=d, columns=self.data.columns) # To change type from npadrray to dataframe
        print(self.data.info())
        print(self.data.head())

        self.data.sample(frac=1)    # To shuffle data

    def cluster(self, n_cluster=4, linkage="ward"):
        """
        To cluster dataset
        :param n_cluster:
        :param linkage:
        :return: array, a list of cluster ids
        """
        self.hc = AgglomerativeClustering(n_clusters=n_cluster, affinity="euclidean", linkage=linkage)  # To define model
        result = self.hc.fit_predict(self.data)     # To cluster

        return result

    def draw_dendrogram(self, method="ward"):
        """
        To draw dendrogram
        :param method: string, method for computing closeness
        :return:
        """
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram (Current Linkage: "+method+")")    # To define figure object
        shc.dendrogram(shc.linkage(self.data, method=method))   # To draw dendrogram for the clustering
        plt.show()

    def draw_relationship(self, target_x="Milk", target_y="Grocery"):
        """
        To draw 2-dimensional scatter considering input two features
        :param target_x: string, feature name applied at x-axis
        :param target_y: string, feature name applied at y-axis
        :return:
        """
        plt.figure(figsize=(10, 10))    # To define figure size
        plt.title("Cluster Scatter between "+target_x+" and "+target_y) # To set figure title
        plt.xlabel(target_x)    # To set label for x axis
        plt.ylabel(target_y)    # To set label for y axis
        plt.scatter(self.data[target_x], self.data[target_y], c=self.hc.labels_)    # To scatter points
        plt.show()  # To draw scatter graph

    def compute_silhouette_score(self, n_cluster, linkage):
        """
        To compute silhouette score considering input # of clusters
        :param n_cluster: int, number of clusters
        :param linkage: string, target linkage
        :return: double, the score
        """
        labels = self.cluster(n_cluster=n_cluster, linkage=linkage) # To cluster labels
        score = silhouette_score(self.data, labels)     # To compute silhouette score
        return score    # To return silhouette score


if __name__ == '__main__':
    wcc = WholesaleCustomerCluster()
    wcc.load_dataset("../dataset/wholesale Customer/Wholesale Customer Dataset.csv")
    wcc.preprocess_dataset()
    for linkage in ["ward", "complete", "average", "single"]:
        wcc.draw_dendrogram(method=linkage)
    # linkage = "ward"
    # print("Current Linkage: ", linkage)
    # print("Current Linkage: ", linkage)
    # wcc.draw_dendrogram(method=linkage)
    # result = wcc.cluster(linkage=linkage)
    # print(wcc.hc.labels_)
    # wcc.data["clusterID"] = wcc.hc.labels_
    # print(wcc.data)
    # wcc.draw_relationship()
    # print("\n\n")
    #
    # for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     score = wcc.compute_silhouette_score(i,linkage)
    #     print("Silhouette Score of", i, " Clusters: ", round(score, 4))
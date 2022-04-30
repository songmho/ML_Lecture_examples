"""
Date: 2020. 09. 09.
Programmer: MH
Description: Code for making customer groups considering purchase items
"""
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from joblib import dump, load
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

    def cluster(self, n_cluster=4, n_init=10, max_iter=300):
        """
        To cluster dataset
        :param n_cluster:
        :return: array, a list of cluster ids
        """
        self.km = KMeans(n_clusters=n_cluster, n_init=n_init, max_iter=max_iter, init="random")  # To define model
        result = self.km.fit_predict(self.data)     # To cluster

        return result

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
        plt.scatter(self.data[target_x], self.data[target_y], c=self.km.labels_)    # To scatter points
        plt.show()  # To draw scatter graph

    def compute_silhouette_score(self, clusters):
        """
        To compute silhouette score considering input # of clusters
        :param clusters: nparray, clustering results
        :return: double, the score
        """
        score = silhouette_score(self.data, clusters)     # To compute silhouette score
        return score    # To return silhouette score

    def save_result(self, path="kmeans_wholesale_cluster.joblib"):
        """
        To save clustering result to local
        :param path: string, path of local file
        :return:
        """
        dump(self.km, path)

    def load_result(self, path="kmeans_wholesale_cluster.joblib"):
        """
        To load clustering result from local
        :param path: string, path of saved file
        :return:
        """
        self.km = load(path)


if __name__ == '__main__':
    wcc = WholesaleCustomerCluster()
    wcc.load_dataset("../dataset/wholesale Customer/Wholesale Customer Dataset.csv")
    wcc.preprocess_dataset()
    print(wcc.data.head(20))
    # clusters = wcc.cluster()
    # print(clusters)
    for i, j, k in [(4, 3, 50),(4, 3, 300),(4, 3, 500),(4, 10, 50),(4, 10, 300),(4, 10, 500),(4, 50, 50),(4, 50, 300),(4, 50, 500)]:
        clusters = wcc.cluster(n_cluster=i, n_init=j, max_iter=k)
        wcc.save_result()
        wcc.load_result()
        print("Silhouette Score of "+str(i)+", "+str(j)+", "+str(k)+": ", wcc.compute_silhouette_score(clusters))

    # for linkage in ["ward", "complete", "average", "single"]:
    #     wcc.draw_dendrogram(method=linkage)
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

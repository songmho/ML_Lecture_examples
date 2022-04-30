"""
Date: 2020. 11. 10.
Programmer: MH
Description: Code for clustering obesity level
"""
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None, "display.max_columns", None)


class ObesityLevelCluster:
    def __init__(self):
        self.threshold = 0.15
        self.list_features = []

    def load_dataset(self, path):
        """
        To load data from local
        :param path: string, path of dataset
        :return:
        """
        self.data = pd.read_csv(path)   # To load dataset
        print(self.data.head())
        print(self.data.info())

    def preprocess_dataset(self):
        """
        To change textual data to numerical data and normalize whole data's instance value from 0 to 1
        :return:
        """
        # To change data format (textual -> Numerical)
        set_freq = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        set_NObesity = {"Insufficient_Weight": 0, "Normal_Weight": 1, "Overweight_Level_I": 2, "Overweight_Level_II": 3, "Obesity_Type_I": 4, "Obesity_Type_II": 5, "Obesity_Type_III": 6}

        ord_encoder = OrdinalEncoder()
        self.data.loc[:, ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", 'SCC', "MTRANS"]] \
            =ord_encoder.fit_transform(self.data.loc[:, ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", 'SCC', "MTRANS"]])

        for k, v in set_freq.items():
            self.data["CAEC"] = self.data["CAEC"].replace(k, v)
            self.data["CALC"] = self.data["CALC"].replace(k, v)

        for k, v in set_NObesity.items():
            self.data["NObesity"] = self.data["NObesity"].replace(k, v)


        data_corr = self.data.corr()        # To compute correlation to "NObesity"
        # print(data_corr.iloc[:, -1])
        # f = plt.figure(figsize=(19, 15))
        # plt.matshow(data_corr, fignum=f.number)
        # plt.xticks(range(self.data.shape[1]), self.data.columns, fontsize=14, rotation=45)
        # plt.yticks(range(self.data.shape[1]), self.data.columns, fontsize=14)
        # cb = plt.colorbar()
        # cb.ax.tick_params(labelsize=14)
        # plt.show()

        self.weighs = list(self.data.loc[:, "Weight"])
        self.height = list(self.data.loc[:, "Height"])
        self.age = list(self.data.loc[:, "Age"])
        self.NObesity = list(self.data.loc[:, "NObesity"])
        self.data.drop("NObesity", 1, inplace=True)
        print(self.data.info())
        # To normalize data size to 0..1
        # mms = MinMaxScaler()
        # d = mms.fit_transform(self.data)    # To normalize data
        # self.data = pd.DataFrame(data=d, columns=self.data.columns)     # To change data type ndarray to dataframe

        # data_corr = self.data.corr()        # To compute correlation to "NObesity"
        # for k, v in data_corr.iloc[:, -1].to_dict().items():    # To select features showing higher corr than threshold
        #     if self.threshold < abs(v):
        #         self.list_features.append(k)
        # self.list_features.remove("NObesity")   # To remove obesity level data
        # self.classes = self.data.loc[:, "NObesity"]
       #  self.data = self.data.loc[:, self.list_features]    # To extract # of features


    def draw_distribution_for_label(self, list_x, list_y, list_z, classes, label_x, label_y, label_z):
        """
        To draw scatter graph considering distribution of input dataset
        :param list_x: list, data for x axis
        :param list_y: list, data for y axis
        :param list_z: list, data for z axis
        :param classes: list, data for label
        :param label_x: string, label for x axis
        :param label_y: string, label for y axis
        :param label_z: string, label for z axis
        :return:
        """
        fig = plt.figure()
        if list_z is not None:
            ax = fig.add_subplot(111, projection='3d')
            dic_data = self.make_set_for_class(list_x,list_y, list_z, classes)  # To make set considering the class ids
            for d in dic_data:  # To draw graph
                ax.scatter(dic_data[d]["x"], dic_data[d]["y"], dic_data[d]["z"], label=d, s=50, alpha=0.5)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.set_zlabel(label_z)
        else:
            ax = fig.add_subplot(111)
            dic_data = self.make_set_for_class(list_x, list_y, list_z=None, classes=classes)  # To make set considering the class ids
            for d in dic_data:  # To draw graph
                ax.scatter(dic_data[d]["x"], dic_data[d]["y"], label=d, s=50, alpha=0.5)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
        ax.legend(loc="lower right")
        ax.legend()
        fig.show()


    def make_set_for_class(self, list_x, list_y,list_z, classes):
        """
        To make a set considering class ids
        :param list_x: list, data for x axis
        :param list_y: list, data for y axis
        :param list_z: list, data for z axis
        :param classes: list, data for each instance's label
        :return:
        """
        result = {}
        ids = sorted(set(classes))
        for i in ids:
            result[i] = {'x':[], 'y':[], "z":[]}

        for i in range(len(list_x)):
            result[classes[i]]['x'].append(list_x[i])
            result[classes[i]]['y'].append(list_y[i])
            if list_z is not None:
                result[classes[i]]['z'].append(list_z[i])

        return result

    def cluster(self, n_cluster, linkage="ward"):
        """
        To cluster dataset
        :param n_cluster: int, the number of clusters
        :param linkage: string, the type of applied linkage
        :return: array, a list of cluster ids
        """
        self.hc = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage=linkage, distance_threshold=2500)  # To generate H.C model
        result = self.hc.fit_predict(self.data) # To predict clusters

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
    linkage = "ward"

    olc = ObesityLevelCluster()
    olc.load_dataset("../dataset/ObesityDataSet_raw_and_data_sinthetic/ObesityDataSet_raw_and_data_sinthetic.csv")
    olc.preprocess_dataset()
    # olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, list_z=olc.age,
    #                                  classes=olc.NObesity, label_x="Weight", label_y="Height", label_z="Age")
    # olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, list_z=None,
    #                                  classes=olc.NObesity, label_x="Weight", label_y="Height", label_z="")


    # olc.draw_dendrogram(method=linkage)
    # for i in range(2, 11):
    #     print("# of Clusters: ", i)
    #     result = olc.cluster(n_cluster=i, linkage=linkage)
    #     for j in range(20):
    #         print(result[j])
    #     # olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, classes=result)
    #     # print("ID".rjust(3, " "), "", "Cluster ID")
    #     # d = olc.data
    #     # d["ClusterID"] = result
    #     # data_corr = d.corr()
    #     # for _, i in data_corr.iloc[:, -1].to_dict().items():
    #     #     print(round(i, 3))
    #     # print(data_corr.iloc[:, -1])
    #     # f = plt.figure(figsize=(19, 15))
    #     # plt.matshow(data_corr, fignum=f.number)
    #     # plt.xticks(range(d.shape[1]), d.columns, fontsize=14, rotation=45)
    #     # plt.yticks(range(d.shape[1]), d.columns, fontsize=14)
    #     # cb = plt.colorbar()
    #     # cb.ax.tick_params(labelsize=14)
    #     # plt.show()
    #     print("\n\n")

    # for i in range(50):
    #     print(str(i+1).rjust(3, " "), "     ", result[i])
    # print(list(result[:200]))
    # for i in range(2, 13):
    #     score = olc.compute_silhouette_score(i, linkage)
    #     print("Silhouette Score of", i, ", Clusters: ", round(score, 4))

    # for i in range(2,10):
    #     result = olc.cluster(n_cluster=i, linkage=linkage)
    #     olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, list_z=None,
    #                                      classes=result, label_x="Weight", label_y="Height", label_z="")

    # result = olc.cluster(n_cluster=7, linkage=linkage)
    # olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, list_z=None,
    #                                  classes=result, label_x="Weight", label_y="Height", label_z="")
    #
    # result = olc.cluster(n_cluster=6, linkage=linkage)
    # olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, list_z=None,
    #                                  classes=result, label_x="Weight", label_y="Height", label_z="")
    #
    # # result = olc.cluster(n_cluster=28, linkage=linkage)
    # # olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, classes=result)
    #
    #
    # result = olc.cluster(n_cluster=2, linkage=linkage)
    # olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, list_z=None,
    #                                  classes=result, label_x="Weight", label_y="Height", label_z="")
    #
    #
    result = olc.cluster(n_cluster=7, linkage=linkage)
    print(np.unique(result))
    olc.draw_distribution_for_label(list_x=olc.weighs, list_y=olc.height, list_z=None,
                                     classes=result, label_x="Weight", label_y="Height", label_z="")

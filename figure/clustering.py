from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

for i in [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25]:
    df = pd.read_csv(r"E:\1. Lab\Daily Results\2021\2111\1110\test.csv")
    kmeans = KMeans(n_clusters=i, random_state=0).fit(df.iloc[:, 1:])
    # print(kmeans.labels_)

    score = silhouette_score(df.iloc[:, 1:], kmeans.labels_, metric="euclidean")
    df = df.iloc[:, 1:]
    df["cluster"] = kmeans.labels_
    print("Silhouette Score  "+str(i)+": %.3f"%score)
    # df.to_csv(r"E:\1. Lab\Daily Results\2021\2111\1110\test_result_"+str(i)+".csv")
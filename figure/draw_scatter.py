import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r"E:\1. Lab\Daily Results\2021\2111\1110\test_result_5.csv")
df_0 = df[df["cluster"]==0]
df_1 = df[df["cluster"]==1]
df_2 = df[df["cluster"]==2]
df_3 = df[df["cluster"]==3]
df_4 = df[df["cluster"]==4]
# df_5 = df[df["cluster"]==5]
# df_6 = df[df["cluster"]==6]
plt.scatter(df_0.iloc[:, 1], df_0.iloc[:, 3], alpha=0.3)
plt.scatter(df_1.iloc[:, 1], df_1.iloc[:, 3], alpha=0.3)
plt.scatter(df_2.iloc[:, 1], df_2.iloc[:, 3], alpha=0.3)
plt.scatter(df_3.iloc[:, 1], df_3.iloc[:, 3], alpha=0.3)
plt.scatter(df_4.iloc[:, 1], df_4.iloc[:, 3], alpha=0.3)
plt.show()


fig = plt.figure()
ax = fig.gca(projection="3d", alpha=0.3)

ax.plot(df_0.iloc[:,2], df_0.iloc[:,3], df_0.iloc[:,1], "o", label="Cluster 0")
ax.plot(df_1.iloc[:,2], df_1.iloc[:,3], df_1.iloc[:,1], "o", label="Cluster 1")
ax.plot(df_2.iloc[:,2], df_2.iloc[:,3], df_2.iloc[:,1], "o", label="Cluster 2")
ax.plot(df_3.iloc[:,2], df_3.iloc[:,3], df_3.iloc[:,1], "o", label="Cluster 3")
ax.plot(df_4.iloc[:,2], df_4.iloc[:,3], df_4.iloc[:,1], "o", label="Cluster 4")
# ax.plot(df_5.iloc[:,2], df_5.iloc[:,3], df_5.iloc[:,1], "o", label="Cluster 5")
# ax.plot(df_6.iloc[:,2], df_6.iloc[:,3], df_6.iloc[:,1], "o", label="Cluster 6")
# ax.scatter(df.iloc[:, 1], df.iloc[:, 3], df.iloc[:, 2], c=df.iloc[:, -1])
plt.show()

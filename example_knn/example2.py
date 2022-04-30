from sklearn.neighbors import KNeighborsRegressor

x = [[5, 45], [5.11, 26], [5.6, 30], [5.9, 34], [4.8, 40],
     [5.8, 36], [5.3, 19], [5.8, 28], [5.5, 23], [5.6, 32],]
y = [77, 47, 55, 59, 72, 60, 40, 60, 45, 58]

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(x, y)
print(knr.predict([[5.5, 38]]))
"""
Date: 2021. 11. 10.
Programmer: MH
Description: Code for Generating data to draw scatter
"""
import random
import pandas as pd
import matplotlib.pyplot as plt


data = {"AppearedObjects": [], "TakenTimeZone": [], "Weather": [], "ImageType": []}
repeat = 300

list_appear_objs = [1,1,   2,2,2,2,2,2,2,2,2,   3,   4,   5,   6,6,6,6,6,6,6,   7,   8,   9, 9,
                    10,10,10,10,10,10,10,  11,  12,  13,13,13,13, 13, 13, 13, 13]
list_taken_time_zone = [1,1,1,1,1,1, 2,2, 3, 4,4,4,4, 5, 6, 7,7,7,7]
list_weather = [1, 1,1,1, 1,1, 2, 3, 4,5, 6,6,6,6,6,6,6,6, 7, 8, 9,9,  10,10,10,10,10, 10, 10, 10]
list_img_type = [1,1,1,2,3,4,4,4,4]

for i in range(repeat):
    data["AppearedObjects"].append(random.choice(list_appear_objs))
    data["TakenTimeZone"].append(random.choice(list_taken_time_zone))
    data["Weather"].append(random.choice(list_weather))
    data["ImageType"].append(random.choice(list_img_type))


df = pd.DataFrame.from_dict(data)
df.to_csv(r"E:\1. Lab\Daily Results\2021\2111\1110\test.csv")

df = pd.read_csv(r"E:\1. Lab\Daily Results\2021\2111\1110\test.csv")
plt.scatter(df.iloc[:, 1], df.iloc[:, 3], alpha=0.3)
plt.show()
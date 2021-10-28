# Importing all important libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Reading the csv file
df = pd.read_csv("star_data_with_gravity.csv")

# Storing mass and radius in different list using iloc method
masses = df.loc[:, "mass"]
radiuses = df.loc[:, "radius"]

# print(masses)
# print(radiuses)

X = []
for index, mass in enumerate(masses):
    temp_list = [radiuses[index], mass]
    X.append(temp_list)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# print(wcss)

plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), wcss, marker = "o", color = "red")
plt.title("The Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

print("By seeing the graph we can conclude that there are three clusters of stars in the data according to their mass and radius")
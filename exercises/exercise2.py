# Exercise 2

# 0. packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 1. Principle Component Analysis

FNAME = "./data/olympics.csv "
df = pd.read_csv(FNAME, index_col=0)
df.info()
df.describe()
df = df.drop(["score"], axis=1)
df

from sklearn.preprocessing import StandardScaler

X = df.values #returns a numpy array
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(pd.DataFrame(X_scaled).describe())

pca = PCA(random_state = 42)
pca.fit(X_scaled)
pca.components_
df_pca = pd.DataFrame(pca.components_, columns=df.columns)
# most prominently  for 1: 110
# most prominently for 2: disq
# for 3: haut

var = pd.DataFrame(pca.explained_variance_ratio_, columns = ["Explained variance"])
var["Cum. explained variance"] = var["Explained variance"].cumsum()
var.plot(kind="bar") # needs 5?

# 2. Clustering
from sklearn.datasets import load_iris
Iris = load_iris()
Iris
X = Iris["data"]
X.shape
y = Iris["target"]
# oder from sklearn.datasets import load_iris
#data_iris = sklearn.datasets.load_iris(return_X_y = False, as_frame = True)
# iris_dataset = data_iris.data


X_scaled = scaler.fit_transform(X)

# k means model
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
kmeans.labels_

df = pd.DataFrame(kmeans.labels_, columns=["kmeans"])

# agglomerative model
from sklearn.cluster import KMeans, AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X_scaled)
df["agg"] = agg.labels_

# DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(min_samples=2, eps=1, metric="euclidean")
dbscan.fit(X_scaled)
df["dbscan"] = dbscan.labels_
df["dbscan"].value_counts()

#scores
from sklearn.metrics import silhouette_score
print(silhouette_score(X, dbscan.labels_)) #best
print(silhouette_score(X, kmeans.labels_))
print(silhouette_score(X, agg.labels_))

df["sepal width"] = X[:,1]
df["petal length"] = X[:,2]

# noise ist -1
df = df.replace(-1, "Noise")

fig, axes = plt.subplots(3, 1)
sns.scatterplot(ax = axes[0], data = df, x = "sepal width",
                y = "petal length", hue = "kmeans")
sns.scatterplot(ax = axes[1], data = df, x = "sepal width",
                y = "petal length", hue = "agg")
sns.scatterplot(ax = axes[2], data = df, x = "sepal width",
                y = "petal length", hue = "dbscan")

fig.savefig("./output/cluster_petal.pdf")

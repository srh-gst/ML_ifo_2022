
# 0. packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# principle component analysis
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
X = boston["data"]
X.shape
y = boston["target"]

from sklearn.decomposition import PCA

pca = PCA(random_state = 42)
pca.fit(X)  # can take very long with large data
pca.components_ # these are the eigenvalues

sns.heatmap(pca.components_, xticklabels = boston["feature_names"]) # feature b and tax important
plt.savefig("./output/heatmap_pca_noscaling.pdf")
var = pd.DataFrame(pca.explained_variance_ratio_, columns = ["Explained variance"])
var["Cum. explained variance"] = var["Explained variance"].cumsum()
var.plot(kind="bar")

df = pd.DataFrame(X, columns = boston["feature_names"])
df.describe()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # scaling usually produces better results
scaler.fit(X)
print(scaler.data_max_)
print(scaler.data_min_)
X_scaled = scaler.transform(X)
scaled = pd.DataFrame(X_scaled).T
unscaled = pd.DataFrame(X).T

fig, axes = plt.subplots(1, 2)
sns.scatterplot(ax = axes[0], data = unscaled, x=0, y=1)
sns.scatterplot(ax = axes[1], data = scaled, x=0, y=0)

# PCA with scaling
pca = PCA(random_state = 42)
pca.fit(X_scaled)
# where are components that tare very light r very dark?
sns.heatmap(pca.components_, xticklabels = boston["feature_names"], cmap = "cubehelix")
plt.savefig("./output/heatmap_pca_scaling.pdf")

var = pd.DataFrame(pca.explained_variance_ratio_, columns = ["Explained variance"])
var["Cum. explained variance"] = var["Explained variance"].cumsum()
var.plot(kind="bar") # first component explains about 55% of the variance. to reach about 90% use maybe 5 components

# clustering algorithms

# k-means clustering

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled) # apparently 42 best random state
kmeans.labels_

df = pd.DataFrame(X, columns=boston["feature_names"])
df["kmeans"] = kmeans.labels_
df.head()
df.groupby("kmeans").mean()

cols = ["INDUS", "AGE", "DIS", "RAD", "TAX", "LSTAT"]
sns.pairplot(df, vars=cols, hue="kmeans")

# Agglomerative Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3, affinity="manhatten", linkage="complete")
agg.fit(X_scaled)
df["agg"] = agg.labels_
df.groupby("agg").mean()

# now for comparison
agg = AgglomerativeClustering(n_clusters=3, affinity="euclid", linkage="complete")
agg.fit(X_scaled)
df["agg_m"] = agg.labels_
df.groupby("agg_m").mean()

sns.pairplot(df, vars=cols, hue="agg_m", diag_kind="hist") # looks worse

agg = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
agg.fit(X_scaled)
df["agg_w"] = agg.labels_
df.groupby("agg_w").mean()

sns.pairplot(df, vars=cols, hue="agg_w", diag_kind="hist") # black all over the place mixed with the beige ones but cl 1 looks good

# direct comparison how often are observations in the cluster based on different clustering methods
pd.crosstab(df["kmeans"], df["agg_w"], margins=True) # we want to have many zeros non-zero is disagreement
pd.crosstab(df["agg"], df["agg_w"], margins=True)

# dbscan
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(min_samples=5, eps=0.65, metric="euclidean")
dbscan.fit(X_scaled)
df["dbscan"] = dbscan.labels_
df["dbscan"].value_counts()

#scores
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
print(silhouette_score(X, dbscan.labels_))
print(silhouette_score(X, kmeans.labels_))
print(silhouette_score(X, agg.labels_)) #best (the higher the better)

print(davies_bouldin_score(X, dbscan.labels_)) # now best
print(davies_bouldin_score(X, kmeans.labels_))
print(davies_bouldin_score(X, agg.labels_))

print(calinski_harabasz_score(X, dbscan.labels_))
print(calinski_harabasz_score(X, kmeans.labels_)) # now best
print(calinski_harabasz_score(X, agg.labels_))
# --> three different measures --> three different results










#import random
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


warnings.filterwarnings('ignore')

# en gros dans ce tp, on va utiliser la méthode kmeans pour grouper les données similaires dans un seul (cluster) (le cluster represente un groupe) (et possede le Centroids comme centre (moyenne))

# lecture de 2019.csv et son transformation en dataframe et puis affichage de quelques informations utiles ainsi que les valeurs manquantes si disponible
data = pd.read_csv("2019.csv") 
print(data.head())
print(data.info())
print(data.isnull().any())
print(data['Country or region'].value_counts())   #la colonne s'appelle Country or region et pas Country

df = data.drop('Country or region', axis=1)
print(df.head())
print(df.isna().sum())
print("somme des duplications: ", df.duplicated().sum())


# Normalizing over the standard deviation
X = df.values[:,1:]                              # extracts the feature values from the dataframe df, start from column 2
X = np.nan_to_num(X)                             # replaces any NaN (Not a Number) values in the array X with numerical values
Clus_dataSet = StandardScaler().fit_transform(X) # apres la standarisation, l'écart type est toujours 1 et la valeur (moyenne) devient plus proche de 0
print(Clus_dataSet)


# Modeling ( on va essayer de configurer et puis visualiser )
# here We used the Silhouette coefficient metric to check if k=5 will result in a good quality of clustering.
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Clus_dataSet)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(Clus_dataSet, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Clusters')
plt.xlabel('Number of clusters (k)')    
plt.ylabel('Silhouette Score')
plt.show()


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(Clus_dataSet) # on a remplacé X par Clus_dataSet pour travailler avec les données apres standarisation

labels = k_means.labels_
print(labels)


# Evaluation du résultat de mon modele Kmeans pour corriger etc...
# Get the cluster centers (Centroids)
centers = k_means.cluster_centers_
print("Cluster Centers:\n", centers)

# Get the inertia (sum of squared distances to the nearest cluster center) ( l'emplacement des colonnes autour du centre )
inertia = k_means.inertia_
print("Inertia:", inertia)

# silhouette score ( la séparation des groupes )
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(Clus_dataSet, labels) # on a remplacé X par Clus_dataSet pour travailler avec les données standarisées
print("Silhouette Score:", silhouette_avg)



# Insights
df["Score"] = labels
print(df.head(5))
print(df["Score"].value_counts())

# Visualiser ces observations en relation avec kmeans
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Score', fontsize=18)
plt.ylabel('Generosity', fontsize=16)
plt.show()



# Data With Filter and Country Index
df_wh=data.filter(['Country or region','Score', 'GDP_Per_Capita','Social support','Health_Life_Expectancy','Freedom to make life choice','Generosity','Perceptions of corruption'])
df_wh=df_wh.set_index('Country or region')
df_wh.plot.barh(stacked=True,  figsize=(10,28))
plt.show()

df_wh_map =df_wh
print(df_wh_map.head()) # par defaut .head prend les 5 premieres lignes
print(df_wh.info())
print(df_wh.isnull().any())
print("nombre de duplications: ", df_wh.duplicated().sum())


# On visualise le résultat final
# Method for K means clustering
def kmeans_Clustering(df_wh,numberOfClusters):
       #Initializing Kmeans.cluster object was imported from sklearn in begining.
       kmeans = KMeans(n_clusters=numberOfClusters)
       # Fitting the input data and getting the cluster labels
       cluster_labels = kmeans.fit_predict(df_wh)
       # Getting the cluster centers
       cluster_centers = kmeans.cluster_centers_
       cluster_centers.shape
       return cluster_labels,cluster_centers

#Plot the cluster
def plot_cluster(labels,centers,df_wh):
    #Getting number of columns
    numOfDimensions = df_wh.columns.size
    #Number of plots required for 5 dimension with 2 dimensions in each plot
    numberOfPlots = int(numOfDimensions/2)
    #Number of rows and columns for subplots
    fig,ax = plt.subplots(numberOfPlots,1, figsize=(10,10))
    for i,j in zip(range(0,numOfDimensions,2),range(0,numberOfPlots)):
         ax[j].scatter(df_wh.iloc[:, i], df_wh.iloc[:, i+1], c=labels, s=50, cmap='viridis')
         ax[j].scatter(centers[:,i], centers[:, i+1], c='black', s=200, alpha=0.5)
         #print(i)
    plt.subplots_adjust(bottom=-0.5, top=1.5)
    plt.show()

labels,centers =kmeans_Clustering(df_wh,3)
plot_cluster(labels,centers,df_wh)





# conclure le score de la silhouette une deuxième fois
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_wh)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(df_wh, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()




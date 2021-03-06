#%% Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("../Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values
#%% dendrogram
import scipy.cluster.hierarchy as sch
def dendrograma(metodo, xl, yl):
    sch.dendrogram(sch.linkage(x, method = metodo))
    plt.title("dendrogram")
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()
dendrograma("ward", "clients", "euclid dist.")
# train
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(x)
print(y_hc)

def MyClustering(n, kmeans, y, titulo, xlabel, ylabel):
    color = ["red", "blue", "green", "cyan", "magenta", "yellow"]
    for i in range(n):
        plt.scatter(x[y == i, 0], x[y == i, 1], s = 100, c = color[i], label = "Cluester {}".format(i+1))
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
MyClustering(5, None, y_hc, "client's cluster", "Annual revenue (in thousands of $) "," Gases score (1-100) ")
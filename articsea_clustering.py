import netCDF4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from dtw import dtw
import sklearn.cluster
import sklearn
import seaborn as sns
sns.set(color_codes=True)

data = netCDF4.Dataset('data/G10010_SIBT1850_v1.1.nc', 'r', format='NETCDF4')

print('Shape latitude: ', data['latitude'][:].shape)
print('Shape longitude: ', data['longitude'][:].shape)
print('Shape time: ', data['time'][:].shape)
print('Shape seaice_conc: ', data['seaice_conc'][:].shape)
print('Shape seaice_source: ', data['seaice_source'][:].shape)

data['seaice_conc'][:].shape
data['longitude'][:]

time = data['time'][:]
longitude = data['longitude'][:]
latitude = data['latitude'][:]
seaice_con = data['seaice_conc'][:]
print(longitude[:5])
print(latitude[:5])
print(seaice_con[0,0,:5])


def compute_clusters(timestep, latitude, longitude, seaice_con):
    X = []
    y = []

    for i in range(len(latitude)):
        for j in range(len(longitude)):
            X += [[seaice_con[timestep, i, j], latitude[i], longitude[j]]]

    X = np.array(X)
    # y = np.array(y)
    # print(X.shape)

    """
    clustering_algorithm = sklearn.cluster.DBSCAN(eps=0.25,
                                min_samples=5,
                                metric='euclidean',
                                algorithm='auto',
                                leaf_size=30,
                                p=None,
                                n_jobs=1)
    """
    clustering_algorithm = sklearn.cluster.SpectralClustering(n_clusters=100, n_jobs=-1)

    return clustering_algorithm.fit_predict(X=X)

# Compute long/lat pairs
s0 = compute_clusters(0, latitude, longitude, seaice_con)
s1 = compute_clusters(1, latitude, longitude, seaice_con)
counter = Counter(s0)
print('# clusters timestep 0: ', max(s0)+1)
counter = Counter(s1)
print('# clusters timestep 1: ', max(s1)+1)
from sklearn.cluster import KMeans

import numpy
import pandas
import time

with open("gassensor.txt", "r") as in_file:
    data = pandas.read_csv(in_file, delim_whitespace=True)
    
    arr = data.to_numpy()

arr = arr.astype(float)

for i in [4, 16, 64, 256]:

    now = time.time()
    k = KMeans(n_clusters=i, algorithm="yinyang", precompute_distances=False)
    k = k.fit(arr)
    yinyang_time = time.time() - now

    print("Yinyang time for {} clusters: {}".format(i, yinyang_time))

    now = time.time()
    k_full = KMeans(n_clusters=i, algorithm="full", precompute_distances=False)
    k_full = k_full.fit(arr)
    full_time = time.time() - now

    print("Standard time for {} clusters: {}".format(i, full_time))

    now = time.time()
    k_elkan = KMeans(n_clusters=i, algorithm="elkan", precompute_distances=False)
    k_elkan = k_elkan.fit(arr)
    elkan_time = time.time() - now

    print("Elkan time for {} clusters: {}".format(i, elkan_time))

    print()

    print("Yinyang speedup over standard for {} clusters: {}".format(i, full_time / yinyang_time))
    print("Yinyang speedup over elkan for {} clusters: {}".format(i, elkan_time / yinyang_time))

    print()

    print("Inertia for Yinyang with {} clusters: {}".format(i, k.inertia_))
    print("Inertia for Standard with {} clusters: {}".format(i, k_full.inertia_))
    print("Inertia for Elkan with {} clusters: {}".format(i, k_elkan.inertia_))

    print()
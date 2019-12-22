# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Ben Seifert
#

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
cimport cython
from cython cimport floating
from libc.math cimport sqrt
from ..metrics import euclidean_distances
cimport libcpp
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool
from cython.operator cimport dereference, postincrement
from cython.parallel import prange

ctypedef np.float64_t DOUBLE
np.import_array()

# C++ struct to represent a cluster
ctypedef struct cluster:
    vector[DOUBLE] center;
    map[size_t, DOUBLE] center_sparse;
    size_t count;
    size_t count_new; 
    size_t count_new_add;
    size_t count_new_sub;
    DOUBLE distChange;
    bool changed;
    size_t label;

# C++ struct to represent a point
ctypedef struct vertex_data:
    vector[DOUBLE] point;
    map[size_t, DOUBLE] point_sparse;
    size_t best_cluster;
    DOUBLE best_distance;
    bool changed;
    size_t best_cluster_old;  
    vector[DOUBLE] lowerbounds;
    DOUBLE upbound;
    bool outofdata;

# Global variables
cdef vector[cluster] CLUSTERS
cdef size_t NUM_CLUSTERS = 0;
cdef size_t NUM_C_CLUSTERS = 10;
cdef size_t MAX_C_ITERATION = 5;
cdef bool IS_SPARSE = False;
cdef vector[DOUBLE] C_CLUSTERS_changemax;

# helper function to compute distance between points
cdef DOUBLE sqr_distance(const vector[DOUBLE]& a, const vector[DOUBLE]& b) nogil:
    cdef DOUBLE total = 0;
    cdef DOUBLE d;
    for i in range(a.size()):
        d = a[i] - b[i];
        total += d * d;
    return sqrt(total);

# Helper function to translate maxmem argument into number of bytes
def translate_maxmem(maxmem):
    # Translate the input string into a float corresponding to the number
    # of bytes requested
    if maxmem:
        size = len(maxmem)
        if maxmem.endswith("TB"):
            return float(maxmem[0:size - 2]) * 1e12
        elif maxmem.endswith("GB"):
            return float(maxmem[0:size - 2]) * 1e9
        elif maxmem.endswith("MB"):
            return float(maxmem[0:size - 2]) * 1e6
        elif maxmem.endswith("KB"):
            return float(maxmem[0:size - 2]) * 1e3
        else:
            return float(maxmem[0:size - 1])
    else:
        return None

# Initializes a vertex
cdef vertex_data vertex_loader() nogil:
    cdef vertex_data v;
    v.best_cluster = -1
    v.best_distance = INFINITY
    v.changed = False
    return v

# Creates a new cluster and initializes it
cdef cluster create_cluster() nogil:
    cdef cluster c;
    c.count = 0
    c.count_new = 0
    c.count_new_add = 0
    c.count_new_sub = 0
    c.changed = True

    return c

# Computes assignments of points to their corresponding clusters
cdef void getassignment(vector[vertex_data] &cvector, vector[cluster] &cclusters):
    cdef DOUBLE di, dbest;
    cdef size_t cbest;

    # Iterate over the clusters
    for i in range(NUM_CLUSTERS):
        # Initialize dbest to infinity and cbest to -1
        dbest = INFINITY
        cbest = -1
        # Iterate over the clusters of clusters
        for j in range(NUM_C_CLUSTERS):
            # Compute distance from this point to the center of each
            # cluster of clusters
            di = sqr_distance(cvector[i].point, cclusters[j].center)

            # Update closest cluster of clusters if less than dbest
            if dbest > di:
                dbest = di
                cbest = j

        cvector[i].best_cluster = cbest

# Recompute center of cluster
cdef void updatecenter(vector[vertex_data] &cvector, vector[cluster] &cclusters):
    cdef vector[DOUBLE] emptycenter;
    cdef DOUBLE d
    emptycenter.resize(cvector[0].point.size());

    for i in range(cvector[0].point.size()):
        emptycenter[i] = 0;

    for i in range(NUM_C_CLUSTERS):
        cclusters[i].center = emptycenter;
        cclusters[i].count = 0;

    for i in range(NUM_CLUSTERS):
        plus_equal_vector(cclusters[cvector[i].best_cluster].center,cvector[i].point);
        cclusters[cvector[i].best_cluster].count  +=1;

    for i in range(NUM_C_CLUSTERS):
        d = cclusters[i].count
        scale_vector(cclusters[i].center, 1.0/d)

# Helper function to add one vector to another
cdef void plus_equal_vector(vector[DOUBLE] &a, vector[DOUBLE] &b) nogil:
    for i in range(a.size()):
        a[i] += b[i]

# Helper function to multiply a vector by a scalar
cdef void scale_vector(vector[DOUBLE] &a, DOUBLE d) nogil:
    for i in range(a.size()):
        a[i] *= d;

# Makes a vertex with no data except a point
cdef vertex_data make_vertex(point):
    cdef vertex_data v
    v.point = point
    return v

# Makes a cluster with no data except a center
cdef cluster make_cluster(center):
    cdef cluster c
    c.center = center
    return c

# Uses traditional K-means to assign initial clusters to overall
# clusters of clusters 
cdef void C_clusters():
    cdef vector[vertex_data] cvector;
    cvector.resize(NUM_CLUSTERS);
    cdef vector[cluster] cclusters;
    cclusters.resize(NUM_C_CLUSTERS)

    global CLUSTERS

    for i in range(NUM_CLUSTERS):
        cvector[i] = make_vertex(CLUSTERS[i].center)

    for i in range(NUM_C_CLUSTERS):
        cclusters[i] = make_cluster(CLUSTERS[i].center)

    for i in range(MAX_C_ITERATION):
        getassignment(cvector, cclusters)
        updatecenter(cvector, cclusters)

    for i in range(NUM_CLUSTERS):
        CLUSTERS[i].label = cvector[i].best_cluster;

# Wrapper on k_means_initialize function to call it in parallel for all
# points
cdef k_means_initialize_wrapper(vector[vertex_data] &graph):
    from cython.parallel import prange
    cdef int i
    cdef int size = graph.size()
    for i in prange(size, nogil=True):
        k_means_initialize(graph[i])

# Computes the best cluster for a point and sets the point's
# lowerbounds and upbound
cdef void k_means_initialize(vertex_data &vertex) nogil:
    
    cdef size_t prev_asg = vertex.best_cluster
    vertex.best_cluster_old = vertex.best_cluster
    cdef DOUBLE best_distance
    cdef size_t best_cluster
    global CLUSTERS
    cdef DOUBLE d

    best_cluster = -1
    best_distance = INFINITY

    vertex.lowerbounds.resize(NUM_C_CLUSTERS)

    for i in range(NUM_C_CLUSTERS):
        vertex.lowerbounds[i] = INFINITY

    for i in range(NUM_CLUSTERS):
        if CLUSTERS[i].center.size() > 0:
            d = sqr_distance(vertex.point, CLUSTERS[i].center)
            if d < best_distance:
                best_distance = d
                best_cluster = i
            else:
                if vertex.lowerbounds[CLUSTERS[i].label] > d:
                    vertex.lowerbounds[CLUSTERS[i].label] = d

    vertex.best_cluster = best_cluster
    vertex.upbound = best_distance
    vertex.outofdata = False
    vertex.changed = (prev_asg != vertex.best_cluster)

# Creates a cluster and initializes its fields
cdef cluster init_cluster() nogil:
    cdef cluster c
    c.count = 0
    c.count_new = 0
    c.count_new_add = 0
    c.count_new_sub = 0
    c.changed = False
    c.label = 0

    return c

# Takes in a vector of points and computes new cluster centers based on the points'
# cluster affiliations
cdef (size_t, vector[cluster]) cluster_center_reducer(vector[vertex_data] &graph) nogil:
    cdef vector[cluster] new_clusters
    cdef size_t num_changed = 0
    new_clusters.resize(NUM_CLUSTERS)

    for k in range(new_clusters.size()):
        new_clusters[k] = init_cluster()

    cdef int j
    cdef int i
    cdef int x
    cdef int size = graph.size()
    cdef int n_clusters = NUM_CLUSTERS

    # Run a thread for each cluster in parallel
    for j in prange(n_clusters, schedule="dynamic", nogil=True):
        # Iterate over the points
        for i in range(graph.size()):
            # If the index of this thread matches the best cluster for this point, operate
            if graph[i].best_cluster % n_clusters == j: 
                if new_clusters[graph[i].best_cluster].count == 0:
                    new_clusters[graph[i].best_cluster].center = graph[i].point
                else:
                    plus_equal_vector(new_clusters[graph[i].best_cluster].center, graph[i].point)
                new_clusters[graph[i].best_cluster].count += 1
                num_changed += graph[i].changed

    # Return the total number of changed points and the new clusters
    return (num_changed, new_clusters)

# Takes in a vector of points and computes changes in cluster centers based on the points' updated
# cluster affiliations and prior cluster affiliations
cdef (size_t, vector[cluster]) cluster_center_reducer_redun(vector[vertex_data] &graph) nogil:
    cdef vector[cluster] new_clusters
    cdef size_t num_changed = 0
    new_clusters.resize(NUM_CLUSTERS)

    for i in range(new_clusters.size()):
        new_clusters[i] = init_cluster()

    cdef int j
    cdef int k
    cdef int x
    cdef int size = graph.size()
    cdef int n_clusters = NUM_CLUSTERS

    # Run a thread for each cluster in parallel
    for j in prange(n_clusters, schedule="dynamic", nogil=True):
        # Iterate over the points
        for k in range(size):
            # If the index of this thread matches the best cluster for this point, operate
            if graph[k].best_cluster % n_clusters == j:
                new_clusters[graph[k].best_cluster].count_new += 1
                new_clusters[graph[k].best_cluster].count_new_add += 1
                if new_clusters[graph[k].best_cluster].center.size() > 0:
                    for x in range(graph[k].point.size()):
                        new_clusters[graph[k].best_cluster].center[x] += graph[k].point[x]
                else:
                    new_clusters[graph[k].best_cluster].center.resize(graph[k].point.size())
                    for x in range(graph[k].point.size()):
                        new_clusters[graph[k].best_cluster].center[x] = graph[k].point[x]

            # If the index of this thread matches the old best cluster for this point, operate
            if graph[k].best_cluster_old % n_clusters == j:
                new_clusters[graph[k].best_cluster_old].count_new += 1
                new_clusters[graph[k].best_cluster_old].count_new_sub += 1
                if new_clusters[graph[k].best_cluster_old].center.size() > 0:
                    for x in range(graph[k].point.size()):
                        new_clusters[graph[k].best_cluster_old].center[x] -= graph[k].point[x]
                else:
                    new_clusters[graph[k].best_cluster_old].center.resize(graph[k].point.size())
                    for x in range(graph[k].point.size()):
                        new_clusters[graph[k].best_cluster_old].center[x] = -1.0 * graph[k].point[x]

    # Return the total number of changed points and the new clusters
    return (size, new_clusters)

# Wrapper on the k_means_iteration_paper function to call it in parallel for
# each point
cdef k_means_paper_wrapper(vector[vertex_data] &graph):
    cdef int i
    cdef int size = graph.size()
    for i in prange(size, nogil=True):
        k_means_iteration_paper(graph[i])

# Recompute the best cluster for each point based on current information
# stored with the point, updated its upbound and lowerbounds
cdef void k_means_iteration_paper(vertex_data &vertex) nogil:
    cdef size_t prev_asg = vertex.best_cluster
    vertex.best_cluster_old = vertex.best_cluster
    global CLUSTERS

    # Step 1: update all group lowerbounds and upbound
    cdef DOUBLE globallowerbound = INFINITY
    cdef vector[DOUBLE] templowerbounds
    templowerbounds.resize(NUM_C_CLUSTERS)

    for i in range(NUM_C_CLUSTERS):
        templowerbounds[i] = vertex.lowerbounds[i]
        vertex.lowerbounds[i] = vertex.lowerbounds[i] - C_CLUSTERS_changemax[i]
        if globallowerbound > vertex.lowerbounds[i]:
            globallowerbound = vertex.lowerbounds[i]
    if CLUSTERS[vertex.best_cluster].distChange > 0:
        vertex.upbound += CLUSTERS[vertex.best_cluster].distChange
        vertex.outofdata = True

    # Step 2: Update point assignment
    cdef bool updateub
    cdef vector[bool] updatewholeornot
    cdef DOUBLE d
    cdef DOUBLE di
    # Filtering 1: this is the "global" filtering
    if vertex.upbound > globallowerbound:
        # Filtering 2: otherwise, prepare for group filtering
        updateub = False
        updatewholeornot.resize(NUM_C_CLUSTERS)

        # mark groups that did not pass the group filtering.
        for i in range(NUM_C_CLUSTERS):
            updatewholeornot[i] = False
            if vertex.upbound > vertex.lowerbounds[i]:
                updateub = True
                updatewholeornot[i] = True
                vertex.lowerbounds[i] = INFINITY

        # update upbound if necessary
        if vertex.outofdata and updateub:
            d = sqr_distance(vertex.point, CLUSTERS[vertex.best_cluster].center)
            vertex.upbound = d
            vertex.outofdata = False

        # anotherway to iterate over all clusters is group by group
        for i in range(NUM_CLUSTERS):
            if (i != prev_asg) and updatewholeornot[CLUSTERS[i].label]:
                if CLUSTERS[i].center.size() > 0:
                    # Filtering 3: left side is the group second best; right side is the point to center lower bound
                    if vertex.lowerbounds[CLUSTERS[i].label] > (templowerbounds[CLUSTERS[i].label] - CLUSTERS[i].distChange):
                        di = sqr_distance(vertex.point, CLUSTERS[i].center)
                        if di < vertex.lowerbounds[CLUSTERS[i].label]:
                            if di < vertex.upbound:
                                vertex.lowerbounds[CLUSTERS[vertex.best_cluster].label] = vertex.upbound
                                vertex.upbound = di
                                vertex.outofdata = False
                                vertex.best_cluster = i
                            else:
                                vertex.lowerbounds[CLUSTERS[i].label] = di

        updatewholeornot.clear()

    templowerbounds.clear()
    vertex.changed = (prev_asg != vertex.best_cluster)

# Main function exposed. Carries out the k-means process
def k_means_yinyang(np.ndarray[floating, ndim=2, mode='c'] X_,
                    int n_clusters,
                    np.ndarray[floating, ndim=2, mode='c'] init,
                    float tol=1e-4, int max_iter=30, verbose=False,
                    maxmem=None):
    global NUM_C_CLUSTERS
    global CLUSTERS
    global NUM_CLUSTERS
    NUM_CLUSTERS = n_clusters
    NUM_C_CLUSTERS = NUM_CLUSTERS / 10

    # Use the memory argument - this is how it is used in the provided code
    memory_size = translate_maxmem(maxmem)
    if memory_size:
        if memory_size/8 < NUM_C_CLUSTERS:
            NUM_C_CLUSTERS = int(memory_size/8)
    if NUM_C_CLUSTERS == 0:
        NUM_C_CLUSTERS = 1

    # Load the input data points into vertex_data objects
    cdef vector[vertex_data] graph;
    graph.resize(len(X_))

    for i in range(len(X_)):
        graph[i] = vertex_loader()
        graph[i].point.resize(len(X_[0]))
        for j in range(len(X_[0])):
            graph[i].point[j] = X_[i,j]

    # Load the initial clusters into cluster objects and initialize
    # the C_CLUSTERS_changemax vector
    CLUSTERS.resize(NUM_CLUSTERS);
    C_CLUSTERS_changemax.resize(NUM_C_CLUSTERS);

    for i in range(n_clusters):
        CLUSTERS[i] = create_cluster()
        CLUSTERS[i].center.resize(len(init[i]))
        for j in range(len(init[i])):
            CLUSTERS[i].center[j] = init[i,j]

    # Map initial clusters to clusters of clusters
    C_clusters()

    # Compute initial cluster assignments for each point
    k_means_initialize_wrapper(graph)

    clusters_changed = True
    iteration_count = 0

    # Loop while the clusters change (until convergence)
    while (clusters_changed):
        # If we have exceeded the max number of iterations, break
        if (max_iter > 0) and (iteration_count >= max_iter):
            break
        
        # If this is the first iteration, behave differently
        if iteration_count == 0:
            # Use the special cluster center calculation method for the first iteration
            num_changed, new_clusters = cluster_center_reducer(graph)

            # Update the global CLUSTERS vector with new cluster data
            for i in range(NUM_CLUSTERS):
                d = new_clusters[i].count
                if d > 0:
                    scale_vector(new_clusters[i].center, 1.0 / d)

                if (new_clusters[i].count == 0) and (CLUSTERS[i].count > 0):
                    print("Cluster {} lost".format(i))
                    CLUSTERS[i].center.clear()
                    CLUSTERS[i].count = 0
                    CLUSTERS[i].changed = False
                else:
                    label = CLUSTERS[i].label
                    CLUSTERS[i] = new_clusters[i]
                    CLUSTERS[i].label = label
                    CLUSTERS[i].changed = True

            # Reset the clusters_changed variable to True    
            clusters_changed = (iteration_count == 0) or (num_changed > 0)

        # Behavior if this is not the first iteration
        else:
            # Figure out if any points changed assignments
            changed_vertices = [v for v in graph if v.changed]

            # Compute changes to clusters as a result of points changing
            # cluster assignments
            num_changed, new_clusters = cluster_center_reducer_redun(changed_vertices)

            # Clear the C_CLUSTERS_changemax vector
            for i in range(NUM_C_CLUSTERS):
                C_CLUSTERS_changemax[i] = 0

            # Update all of the clusters based on changed cluster assignments from the last
            # iteration
            for i in range(NUM_CLUSTERS):
                new_clusters[i].count = CLUSTERS[i].count + new_clusters[i].count_new_add - new_clusters[i].count_new_sub

                d = new_clusters[i].count;
                d1 = new_clusters[i].count_new;

                if d > 0:
                    if d1 > 0:
                        center_temp = CLUSTERS[i].center
                        scale_vector(CLUSTERS[i].center, CLUSTERS[i].count)
                        plus_equal_vector(new_clusters[i].center, CLUSTERS[i].center)
                        scale_vector(new_clusters[i].center, 1.0 / d)
                        d_update = sqr_distance(new_clusters[i].center,center_temp)
                        if C_CLUSTERS_changemax[CLUSTERS[i].label] < d_update:
                            C_CLUSTERS_changemax[CLUSTERS[i].label] = d_update
                        label = CLUSTERS[i].label
                        CLUSTERS[i] = new_clusters[i]
                        CLUSTERS[i].label = label
                        CLUSTERS[i].distChange = d_update
                    else:
                        CLUSTERS[i].distChange = 0

                if (d == 0) and (CLUSTERS[i].count > 0):
                    print("Cluster {} lost".format(i))
                    print("Cluster {} lastrun {}".format(i, CLUSTERS[i].count))
                    print("Cluster {} d1 {}".format(i, new_clusters[i].count_new))
                    CLUSTERS[i].center.clear();
                    CLUSTERS[i].count = 0;
                    CLUSTERS[i].changed = False;
                    CLUSTERS[i].distChange = 0;
                elif d1 == 0:
                    CLUSTERS[i].distChange = 0;
                    CLUSTERS[i].changed = False;

            # Compute whether or not the clusters actually changed
            clusters_changed = (iteration_count == 0) or (num_changed > 0)

        # If this is the first iteration, call the k_means_initialize_wrapper
        # to reassign points to clusters
        if iteration_count == 0:
            k_means_initialize_wrapper(graph)
        # otherwise, use the new Yinyang K-means method of computing
        # new assignments of points to clusters
        else:
            k_means_paper_wrapper(graph)

        # Increment the iteration count
        iteration_count += 1

    # Prepare the lists of labels and centroids to return
    labels = np.array([v.best_cluster for v in graph])
    centroids = np.array([c.center for c in CLUSTERS])

    return centroids, labels, iteration_count

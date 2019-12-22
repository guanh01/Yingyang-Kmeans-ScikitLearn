## Project: Extend Scikit-Learn with Yinyang K-Means 

## Project Description:
Scikit-Learn is an open-source Machine Learning library for Python, widely used. The goal of this project is to extend Scikit-Learn by adding Yinyang K-Means into it as an alternative to the KMeans algorithm in Scikit-Learn.
Yinyang K-Means is an optimized KMeans algorithm. By clearly applying triangle inequality, it effectively avoids a large portion of the distance computations in the traditional KMeans. It gives significant speedups without compromising the clustering quality. Details are shown in this paper (https://people.engr.ncsu.edu/xshen5/Publications/icml15.pdf).

A version of Yinyang K-Means implemented by the authors of the research paper is available here (http://research.csc.ncsu.edu/nc-caps/yykmeans.tar.bz2).

## Comments of this Repo:
* The implementation of yingyang k-means in scikit-learn gives an average 2-3X speedup when k is large (e.g., 64). When k is small (e.g., 4 or 16), it is even slower than the baselines.
* Only tested on two small datasets.  
* Refer to README.txt for details on how to install the implementation. 

## Project Requirement
1) Create a local branch of the git repository of scikit-learn, in which, you can develop your code by adding Yinyang K-Means into the class sklearn.cluster.KMeans in Scikit-Learn. There are two changes to the interface of the class:
   a) the argument "algorithm" now has an option "yinyang", which invokes the Yinyang K-Means algorithm.
   b) a new optional argument "maxmem" can be used to indicate the maximal memory Yinyang K-Means is allowed to use; the value can be in the unit of "B", "KB", "MB", "GB", or "TB". An example is:

      sklearn.cluster.KMeans(n_clusters=8, init=’k-means++’, ..., maxmem='2.5GB')

   This option is ignored if the "algorithm" argument is not "yinyang".
   Try to follow a good coding style. The best submission could be suggested to commit to scikit-learn for the community to use.

2) Try to make the implementation of Yinyang K-Means as fast as possible. Several hints:
   a) Python allows the invocation of C libraries (e.g., through
   ctypes). [C extension interface (https://docs.python.org/2/extending/extending.html) offered by CPython is another
   option, although it is not as portable as ctypes.]
   b) If you can make use of multicores and the vector units, they are often quite helpful.

3) Design and run experiments to compare the performance of the "full", "elkan", and "yinyang" algorithms on the datasets used in the Yinyang K-Means paper. Note that any performance comparison has to first make sure that the execution results from the algorithms are equivalent. Due to many randomness and floating-point errors, the clustering results could differ. One way to check the equivalence is to check the clustering quality (see clustering quality measures on this webpage: https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment).
4) Create the documentations for the extended functionality of sklearn.cluster.KMeans.
5) Write the final report.

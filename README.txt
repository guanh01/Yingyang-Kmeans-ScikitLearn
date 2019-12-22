Course Project 1: Yinyang K-means in Scikit-Learn
Ben Seifert
CSC412

Software Requirements:
    Compiler with OpenMP support
    Python (version >= 3.5)
    Pip (corresponding to the version of Python)
    NumPy (version >= 1.11)
    SciPy (version >= 0.17)
    Joblib (version >= 0.11)
    Cython (version >= 0.28.5)
    Pandas (version >= 0.25.3)
    A C/C++ compiler and matching OpenMP runtime library. Platform-specific instructions are detailed at the link below

The requirements and instructions for installation are detailed at this link: https://scikit-learn.org/stable/developers/advanced_installation.html
Specifically, the instructions under the heading "Building from source" are the instructions to be followed.

To install/run my code, first fork the Scikit-Learn project and clone it locally. Then, navigate into the 
scikit-learn/sklearn/cluster directory. Copy the "_k_means.py", "_k_means_yinyang.pyx", and "setup.py" files
that I have provided as part of this .tar file into the scikit-learn/sklearn/cluster directory. This will require overwriting
the existing "_k_means.py" and "setup.py" files in that directory. If prompted to confirm the overwrite, confirm. Then, navigate 
to the top-level directory for the Git project (the scikit-learn directory) and run the command "pip install --editable ."
These instructions assume that all software requirements detailed above are met.

Once the above commands have been executed successfully, you may manually test the implemented functionality, or run one of my test scripts.

I have provided two test scripts, which test my implementation of Yinyang K-means and compare it to the existing implementations
of traditional K-means and Elkan K-means. The test scripts are named "kmeans_gassensor.py" and "k_means_kegg.py". 
Instructions to run them are as follows:

To run the kmeans_gassensor.py script, make sure that Scikit-Learn is installed including my implementation of Yinyang K-means.
Make sure that the "gassensor.txt" file I have provided is in the same directory as the kmeans_gassensor.py script, and make sure
that you are also in that directory as your current working directory. Run the command "python kmeans_gassensor.py". This command
will print output to the terminal detailing the results for running the three K-means algorithms on the gassensor dataset for 
k = 4, 16, 64, and 256.

To run the kmeans_kegg.py script, make sure that Scikit-Learn is installed including my implementation of Yinyang K-means.
Make sure that the "kegg_shuffled_normal.txt" file I have provided is in the same directory as the kmeans_kegg.py script, and make sure
that you are also in that directory as your current working directory. Run the command "python kmeans_kegg.py". This command
will print output to the terminal detailing the results for running the three K-means algorithms on the kegg dataset for 
k = 4, 16, 64, and 256.

Each of the test scripts will output information about total time for each algorithm to perform its clustering on the input dataset,
as well as the speedup of Yinyang K-means over the other two algorithms for each given number of clusters, and the best inertia of the final
clustering. This value can be used to compare the cluster quality produced by the three algorithms to make sure results are comparable.

To manually test Yinyang K-means, import KMeans from sklearn.cluster and build/fit a KMeans object with algorithm="yinyang".

NOTE: If Python or pip are referred to under a different name/alias than "python" or "pip" on the machines where this program is tested, 
it may be necessary to replace "python" or "pip" in the commands given above with the appropriate command to reference Python >= 3.5 and 
Pip that corresponds to that version of Python >= 3.5.

NOTE: LIMITATIONS: There are no known bugs or limitations, except that my implementation of Yinyang K-means does not support sparse data.
Any data passed to the my implementation of the Yinyang K-means algorithm must be dense data.
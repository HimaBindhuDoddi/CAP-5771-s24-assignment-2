import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(input_data, num_clusters):
    
    features, targets = input_data
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=num_clusters, init='random', random_state=42)
    kmeans.fit(scaled_features, targets)
    
    predicted_labels = kmeans.predict(scaled_features)
    
    return predicted_labels

def fit_kmeans_d(input_data, num_clusters):
    
    features, targets = input_data
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=num_clusters, init='random')
    kmeans.fit(scaled_features, targets)
    
    predicted_labels = kmeans.predict(scaled_features)
    
    return predicted_labels
    


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    
    seed = 42
    n_samples = 100
    dct = answers["1A: datasets"] = {}
    nc = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
    dct["nc"] = [nc]
    
    nm = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    dct["nm"] = [nm]
    
    bvv = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    dct["bvv"] =[bvv]
    
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)
    dct["add"] =[add]
    
    b = datasets.make_blobs(n_samples=n_samples, random_state=seed)
    dct["b"] =[b]
     
    
   

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    
    dataset = {
    "nc": nc,
    "nm": nm,
    "bvv": bvv,
    "add": add,
    "b": b
    }

    cluster_counts = [2, 3, 5, 10]
    dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    pdf_filename = "cluster_report_1C.pdf"
    pdf_pages = []
    
    successful_clusters = {}
    failed_clusters = []
    
    fig, axes = plt.subplots(len(cluster_counts), len(dataset_keys), figsize=(20, 16))
    fig.suptitle('Scatter plots for different datasets and number of clusters', fontsize=16)
    
    for j, key in enumerate(dataset_keys):
        data, true_labels = dataset[key]
    
        for i, cluster_count in enumerate(cluster_counts):
            predicted_labels = fit_kmeans(dataset[key],num_clusters=cluster_count)
    
            ax = axes[i, j]
            ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
            ax.set_title(f'{key}, k={cluster_count}')
    
    plt.tight_layout()
    pdf_pages.append(fig)
    plt.close(fig)
    
    with PdfPages(pdf_filename) as pdf:
        for page in pdf_pages:
            pdf.savefig(page)
    

    
    dct = answers["1C: cluster successes"] = {"bvv": [3], "add": [3],"b":[3]}

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["nc","nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
    dataset = {
    "nc": nc,
    "nm": nm,
    "bvv": bvv,
    "add": add,
    "b": b
    }
    cluster_counts = [2, 3]
    dataset_keys = ['nc', 'nm', 'bvv', 'add', 'b']
    
    for ii in range(5):
        successful_clusters = {}
        failed_clusters = []
        pdf_filename = f"cluster_report_1D_{ii+1}.pdf"
        pdf_pages = []

        fig, axes = plt.subplots(len(cluster_counts), len(dataset_keys), figsize=(20, 16))
        fig.suptitle('Scatter plots for different datasets and number of clusters', fontsize=16)

        for j, key in enumerate(dataset_keys):
            data, true_labels = dataset[key]

            for i, cluster_count in enumerate(cluster_counts):
                predicted_labels = fit_kmeans_d(dataset[key], num_clusters=cluster_count)

                ax = axes[i, j]
                ax.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
                ax.set_title(f'{key}, k={cluster_count}')

        plt.tight_layout()
        pdf_pages.append(fig)
        plt.close(fig)

        with PdfPages(pdf_filename) as pdf:
            for page in pdf_pages:
                pdf.savefig(page)


    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = ["nc"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)

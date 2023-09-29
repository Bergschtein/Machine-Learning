import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
import copy


class KMeans:
    
    def __init__(self, k = 2, n_init = 100, maxIterations = 1000):
        """
        Parameters: 
                    k: Number of clusters
                    n_init: Number of times the algorithm is run, best run is selected.
                    maxIterations: Maximal number of iterations for each run. Safety measure.
        """
        self.centroids = None
        self.centroidList = []
        self.lables = None
        self.lablesList = []

        self.maxIterations = maxIterations
        self.k = k
        self.n_init = n_init
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Better initialization
        
        #To np array for easier calculations 
        X = X.to_numpy()

        #Several runs to minimize chances og local minima.
        for i in range(self.n_init):

            #Initialize cluster centroids
            #Sample k random points as initial centroids.
            sample = np.random.choice(X.shape[0], size=self.k, replace=False)
            centroids = X[sample]
            

            #Book keeping vars. For termination
            oldCentroids = np.zeros_like(centroids)
            iterations = 0

            while not self.shouldStop(centroids, oldCentroids, iterations):
                #Need deep copy to disentangle old and new
                oldCentroids = copy.deepcopy(centroids)
                iterations += 1
                
                #Assigns lables 
                distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
                lables = np.array([np.argmin(i) for i in distances])
                
                #Update centroids
                for i in range(len(centroids)):
                    mu = np.mean(X[lables == i], axis = 0)
                    centroids[i] = mu
                
            # print(iterations)

            if iterations > 0:
                #Adds fit to class variables
                self.centroidList.append(centroids)
                self.lablesList.append(lables)

        #Selects best fit based on euclidean distortion
        distortion = []
        for i, lables in enumerate(self.lablesList):
            distortion.append(euclidean_distortion(X, self.lablesList[i]))
        best_run = np.argmin(distortion)
        self.centroids = self.centroidList[best_run]
        self.lables = self.lablesList[best_run]

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = X.to_numpy()
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        lables = np.array([np.argmin(i) for i in distances])
        return lables.astype(int)

    def get_centroidList(self):
        return self.centroidList

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    # Stopping criterion
    def shouldStop(self,centroids, oldCentroids, iterations):
        if iterations > self.maxIterations: return True
        return (oldCentroids == centroids).all()
    
    
    

    
# --- Some utility functions 




def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)

    for i, c in enumerate(clusters):
        # for i in range(len(centroids)):
        #         mu = np.mean(X[lables == i], axis = 0)
        #         centroids[i] = mu
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += (euclidean_distance(Xc, mu)).sum(axis=0)
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  
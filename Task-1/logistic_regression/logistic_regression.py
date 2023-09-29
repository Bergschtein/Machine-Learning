import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:

    def __init__(self, learning_rate = 0.05, epochs = 10000, poly = False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.theta_0 = None
        self.poly = poly

    def fit(self, data, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels
        """

        """
        - Initialize parameters (theta), learng rate,
        - Maximize log-likelihood with gradient descent
        -
        """
        #Expanding the feature space

        #In order to supress warnings.
        X = data.copy(deep = True)
        if self.poly:
            X["x0^2"] = X["x0"]**2
            X["x1^2"] = X["x1"]**2
            X["x0*x1"] = X["x0"]* X["x1"]
            
        X = np.asarray(X)
        #Include constant term  
        X = np.hstack((X, np.ones([X.shape[0], 1], X.dtype)))
        #Initialize parameters
        number_of_features = X.shape[1]
        theta = np.zeros(number_of_features)

        for i in range(self.epochs):
            #Get predictions
            Z = np.dot(theta.T, X.T) #+ theta_0
            y_pred = sigmoid(Z)

            # Update parameters
            theta = theta + self.learning_rate * np.dot((y - y_pred), X)

        self.theta = theta



    def predict(self, data):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        if (self.theta).all() != None:
            #In order to supress warnings.
            if self.poly:
                X = data.copy(deep = True)
                X["x0^2"] = X["x0"]**2
                X["x1^2"] = X["x1"]**2
                X["x0*x1"] = X["x0"]* X["x1"]
            else: X = data
            X = np.asarray(X)
            #Include constant term  
            X = np.hstack((X, np.ones([X.shape[0], 1], X.dtype)))
            return sigmoid(np.dot(X, self.theta) ) #+ self.theta_0




# --- Some utility functions

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input
    """
    return 1. / (1. + np.exp(-x))


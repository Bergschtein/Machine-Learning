import numpy as np 
import pandas as pd 


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class Node:

    def __init(self, )



class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        #Need a graph structure or something
        self.root = None

    

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # TODO: Implement 
        # attributes = X.columns.tolist()
        # unique_target = y.unique()
        # if len(unique_target) == 1:
        #     self.add_root(unique_target)
        #     return # Should i return something??
        
        # if len(attributes) == 0:
        #     mcv = y.value_counts().idxmax() #Most common value
        #     self.add_root(mcv)
        #     return # Should i return something??

        # A = information_gain(X,y)

        # self.add_root
        





    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        pass
        #raise NotImplementedError()
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        pass
        #raise NotImplementedError()


# --- Some utility functions 



def best_node(attributes):
    #How to select the attribute with most information gain
    pass





def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a length k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



def information_gain(X, y):
    """
    Input: Collection of training examples X and target attribute y.
    Output: The attribute corresponding to the highest information gain.

    """
    counts_y = y.value_counts() #S
    entropy_y = entropy(counts_y) #Entropy(S)

    attributes = X.columns.tolist() #All attributes of X
    inf_gain = {}
    for A in attributes:
        values_A = X[A].unique() #All unique values of attribute A
        sum_entropy_y_v = 0
        for v in values_A:
            mask = X[A] == v

            sum_entropy_y_v += (len(y[mask]) / len(y)) * entropy(y[mask].value_counts())
        inf_gain.update({A : entropy_y - sum_entropy_y_v})
    
    max_inf_gain = max(inf_gain, key = inf_gain.get) #Extracts the attribute corresponding to the highest inf. gain.
    return max_inf_gain
    

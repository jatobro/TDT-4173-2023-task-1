import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, max_iter=1000, epsilon=1e-15):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.b = None
        self.w = None
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        def dldb(sigmoid):
            return (sigmoid-y).mean(axis = 0)

        def dldw(sigmoid):
            return ((sigmoid-y).values.reshape((len(X), 1))*X).mean(axis = 0)
        
        def update(a, g):
            return a - (g * self.learning_rate)
        
        def has_converged(b, b_old, w, w_old):
            if np.linalg.norm(b_old) != 0 and np.linalg.norm(w_old) != 0:
                return np.linalg.norm(b - b_old) / np.linalg.norm(b_old) < self.epsilon and np.linalg.norm(w - w_old) / np.linalg.norm(w_old) < self.epsilon
            return False

        
        
        # initialize u
        self.b = 0
        self.w = np.zeros(X.shape[1])

        for _ in range(self.max_iter):
            y_h = self.predict(X)
            sig = sigmoid(y_h)

            gradient_b = dldb(sig)
            gradient_w = dldw(sig)

            b_old = self.b
            w_old = self.w
            
            self.b = update(self.b, gradient_b)
            self.w = update(self.w, gradient_w)

            if has_converged(self.b, b_old, self.w, w_old):
                break
           

    
    def predict(self, X):
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
        return np.matmul(self.w, X.T) + self.b
        

        
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

        
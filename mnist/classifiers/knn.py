"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
from classifiers.utils import utils
from sklearn.base import BaseEstimator, ClassifierMixin

class KNN(BaseEstimator, ClassifierMixin):

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the trianing data
        self.y = y

    def predict(self, Xtest):

        # Note for args in euclidean distance calculation:
        # we want the output array to have same num rows as
        # Xtest since that'll be the basis of our prediction
        dist = utils.euclidean_dist_squared(Xtest, self.X)

        # sort distance in ascending order, for each Xtest data point
        asc_indices = np.argsort(dist)

        res_dimension = asc_indices.shape[0]
        y_pred = np.zeros(res_dimension)

        for i in range(res_dimension):
            # get k nearest neighbours
            k_top_indices = asc_indices[i, :self.k]

            # get corresponding y values for these k neighbours
            y_values = np.zeros(self.k)
            for counter, index in enumerate(k_top_indices):
                y_values[counter] = self.y[index]

            # prediction is the most common y label for the k neighbours
            y_pred[i] = utils.mode(y_values)

        return y_pred

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
import os.path
import scipy as sp
from scipy import stats
from scipy.optimize import approx_fprime
import numpy as np
import matplotlib.pyplot as plt

# KNN helpers
def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """
    # add extra dimensions so that the function still works for X and/or Xtest are 1-D arrays.
    if X.ndim == 1:
        X = X[None]
    if Xtest.ndim == 1:
        Xtest = Xtest[None]

    return np.sum(X ** 2, axis=1)[:, None] + np.sum(Xtest ** 2, axis=1)[None] - 2 * np.dot(X, Xtest.T)

# multi-class linear_classifer helpers (softmax + SVM)
def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma

def check_gradient(model, X, y):

    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w,X,y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-2):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
             (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')

def classification_error(y, yhat):
    return np.mean(y!=yhat)


# neural net
# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b)
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights, ())])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        b_size = layer_sizes[i + 1]

        W = np.reshape(weights_flat[counter:counter + W_size], (layer_sizes[i + 1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter + b_size][None]
        counter += b_size

        weights.append((W, b))
    return weights


def log_sum_exp(Z):
    Z_max = np.max(Z, axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:, None]), axis=1))  # per-column max
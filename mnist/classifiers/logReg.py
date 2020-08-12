import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime
from classifiers.utils import utils
from classifiers.utils import findMin

# non-kernelized classifiers
class logReg:
    """ naive logistic regression class
        loss: logistic loss
        regularization: none
    """

    def __init__(self, verbose=2, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w).asType

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X @ self.w)


class multiLogRegL2(logReg):
    """ multi-class logistic regression
        loss: softmax loss
        regularization: L2
        hyperparameter detail:
            - lam (regularization strength): 1.0 by default
    """

    def __init__(self, lam=1.0, verbose=1, maxEvals=400):
        super().__init__(verbose, maxEvals)
        self.lam = lam

    def funObj(self, w, X, y):
        # dimensions
        n, d = X.shape
        k = np.unique(y).size
        W = w.reshape((k, d))  # reshape vector w into matrix

        # compute softmax loss and add regularization
        exp_XWT = np.exp(X.dot(W.T)).astype('float128')  # n*k (prevent overflow)
        probs = exp_XWT / np.sum(exp_XWT, axis=1, keepdims=True)  # n*k
        true_class_prob = probs[range(n), y]  # n*1
        f = np.sum(-np.log(true_class_prob))  # scalar (loss)
        reg = (self.lam / 2.) * np.sum(W ** 2)  # L2 regularizer
        f += reg

        # compute gradient in matrix form
        probs[range(n), y] -= 1  # the true class probs - indicator function
        g = probs.T.dot(X) + self.lam * W  # k*d

        # vectorize gradient
        g = g.flatten()

    def fit(self, X, y):
        # dimensions
        n, d = X.shape
        self.n_classes = np.unique(y).size  # k
        W_shape = (self.n_classes, d)

        # Initial guess
        self.w = np.zeros(W_shape)  # k*d
        vectorized_W = self.w.flatten()

        # optimize
        utils.check_gradient(self, X, y)
        (vectorized_W, f) = findMin.findMin(self.funObj, vectorized_W,
                                            self.maxEvals, X, y, verbose=self.verbose)
        self.w = vectorized_W.reshape(W_shape)

    def predict(self, X):
        return np.argmax(X @ self.w.T, axis=1)

import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime
from classifiers.utils import utils
from classifiers.utils import findMin

class multiSVML2:
    """ multi-class linear SVM classifier
            loss: Hinge loss using sum rule (mentioned in lecture slide: L20, pp33)
            regularization: L2
            hyperparameter detail: following lambda convention
    """

    def __init__(self, lam=1.0, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.lam = lam

    def funObj(self, w, X, y):
        # dimensions
        n, d = X.shape
        k = np.unique(y).size
        W = w.reshape((k, d))  # reshape vector w into matrix
        delta = 1.0  # loss tolerance for a single example + class combination

        # compute hinge loss and add regularization
        XW = X.dot(W.T)  # (n*k)
        margin_matrix = np.maximum(0, XW[range(n), :] - XW[range(n), y].reshape(-1, 1) + delta)  # (n*k)
        margin_matrix[range(n), y] = 0  # margin for correct class should be 0 (n*k)
        f = np.sum(margin_matrix)  # loss (scalar)
        f += (self.lam / 2.) * np.sum(W ** 2)  # loss with L2-regularizer (scalar)

        # compute gradient in matrix form
        margin_matrix[margin_matrix > 0] = 1  # incorrect classes should have g = 1
        margin_matrix[range(n), y] -= margin_matrix.sum(axis=1)  # correct row should have g = - (# incorrect classes)
        g = (margin_matrix.T).dot(X)  # (k*d)
        g += self.lam * W

        # vectorize gradient
        g = g.flatten()

        return f, g

    def fit(self, X, y):
        # dimensions
        n, d = X.shape
        self.n_classes = np.unique(y).size  # k
        W_shape = (self.n_classes, d)

        # Initial guess
        self.w = np.zeros(W_shape)  # k*d
        vectorized_W = self.w.flatten() # (k*d) * 1

        # optimize
        utils.check_gradient(self, X, y)
        (vectorized_W, f) = findMin.findMin(self.funObj, vectorized_W,
                                            self.maxEvals, X, y, verbose=self.verbose)
        self.w = vectorized_W.reshape(W_shape)

    def predict(self, X):
        return np.argmax(X @ self.w.T, axis=1)


# kernel methods
def kernel_RBF(X1, X2, sigma=0.5):
    n1,d1 = X1.shape
    n2,d2 = X2.shape
    K = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            K[i,j] = np.exp(-np.linalg.norm(X1[i]-X2[j])**2/(2*sigma**2))
    return K

def kernel_poly(X1, X2, p=2, coef=1):
    return (coef+X1@X2.T)**p

class kernelSVML2:
    """ multi-class kernel SVM classifier
        loss: Hinge loss using sum rule (mentioned in lecture slide: L20, pp33)
        regularization: L2
        kernel: polynomial or Gaussian RBF
        hyperparameter detail: following lambda convention
    """
    def __init__(self, lam=1.0, verbose=0, maxEvals=100, kernel_fun=kernel_poly, **kernel_args):
        self.lam = lam
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.kernel_fun = kernel_fun
        self.kernel_args = kernel_args
        self.bias = True

    def funObj(self, w, K, y):
        """ Parameters
            ----------
            w: an N*k by 1 weight vector (k: number of classes)
            K: an N by N gram matrix
            y: an N by 1 target vector
        """
        # dimensions
        n, _ = K.shape
        k = np.unique(y).size
        W = w.reshape((k, n))  # reshape vector w into matrix
        delta = 1.0  # loss tolerance for a single example + class combination

        # compute hinge loss and add regularization
        KW = K.dot(W.T)  # (n*k)
        margin_matrix = np.maximum(0, KW[range(n), :] - KW[range(n), y].reshape(-1, 1) + delta)  # (n*k)
        margin_matrix[range(n), y] = 0  # margin for correct class should be 0 (n*k)
        f = np.sum(margin_matrix)  # loss (scalar)
        f += (self.lam / 2.) * np.sum(W ** 2)  # loss with L2-regularizer (scalar)

        # compute gradient in matrix form
        margin_matrix[margin_matrix > 0] = 1  # incorrect classes should have g = 1
        margin_matrix[range(n), y] -= margin_matrix.sum(axis=1)  # correct row should have g = - (# incorrect classes)
        g = (margin_matrix.T).dot(K)  # (k*n)
        g += self.lam * W

        # vectorize gradient
        g = g.flatten()

        return f, g

    def fit(self, X, y):
        self.X = X
        # dimensions
        n, d = X.shape
        self.n_classes = np.unique(y).size  # k
        W_shape = (self.n_classes, n)

        # Initial guess
        self.w = np.zeros(W_shape)  # k*d
        vectorized_W = self.w.flatten() # (k*d) * 1

        # transform X into gram matrix K
        K = self.kernel_fun(X, X, **self.kernel_args) # n*n

        # optimize
        utils.check_gradient(self, K, y)
        (vectorized_W, f) = findMin.findMin(self.funObj, vectorized_W,
                                            self.maxEvals, K, y, verbose=self.verbose)
        self.w = vectorized_W.reshape(W_shape)

    def predict(self, X):
        Ktest = self.kernel_fun(X, self.X, **self.kernel_args)
        return np.argmax(Ktest @ self.w.T, axis=1)
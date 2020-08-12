import numpy as np
from classifiers.utils import utils
from classifiers.utils import findMin
from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin

class mlp(BaseEstimator, ClassifierMixin):
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes=64, lammy=1, max_iter=100, verbose=1, alpha=0.0001, batch_size=500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.max_iter = max_iter
        self.verbose = verbose
        self.alpha = alpha # learning rate
        self.batch_size = batch_size

    def funObj(self, weights_flat, X, y):
        weights = utils.unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        for W, b in weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
            activations.append(X)

        yhat = Z

        if self.classification:  # softmax
            tmp = np.sum(np.exp(yhat), axis=1)
            # f = -np.sum(yhat[y.astype(bool)] - np.log(tmp))
            f = -np.sum(yhat[y.astype(bool)] - utils.log_sum_exp(yhat))
            grad = np.exp(yhat) / tmp[:, None] - y
        else:  # L2 loss
            f = 0.5 * np.sum((yhat - y) ** 2)
            grad = yhat - y  # gradient for L2 loss

        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W, b = weights[i]
            grad = grad @ W
            grad = grad * (activations[i] * (1 - activations[i]))  # gradient of logistic loss
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)

            g = [(grad_W, grad_b)] + g  # insert to start of list

        g = utils.flatten_weights(g)

        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat ** 2)
        g += self.lammy * weights_flat

        return f, g

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, None]

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1] > 1  # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            b = scale * np.random.randn(1, self.layer_sizes[i + 1])
            weights.append((W, b))
        weights_flat = utils.flatten_weights(weights)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        weights_flat_new, f = findMin.findMin(self.funObj, self.alpha, weights_flat, self.max_iter, X, y, verbose=2)

        self.weights = utils.unflatten_weights(weights_flat_new, self.layer_sizes)

    def fitWithSGD(self, X, y):
        optTol = 1e-2
        if y.ndim == 1:
            y = y[:, None]
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1] > 1  # assume it's classification iff y has more than 1 column

        # random initilize
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes) - 1):
            W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
            b = scale * np.random.randn(1, self.layer_sizes[i + 1])
            weights.append((W, b))
        weights_flat = utils.flatten_weights(weights)

        # SGD:
        # outer loop: iterate through entire X 10 times
        # inner loop: iterate through 100 randomly partitioned mini-batches
        # - perform gradient descent
        # - update weights
        num_batches = X.shape[0] / self.batch_size
        batch_row_ind = np.random.permutation(np.arange(X.shape[0])).reshape(int(num_batches), -1)
        alpha = self.alpha

        # main loop of SGD
        for i in range(self.max_iter):  # 10 iterations
            # randomly partition X
            # each row of batch_row_ind represents row indices of a batch in X
            for k, j in enumerate(batch_row_ind):  # 100 iterations
                # gradient descent on one batch (incl. averaging)
                f, g = self.funObj(weights_flat, X[j], y[j])
                weights_flat = weights_flat - alpha * g
            optCond = norm(g, float('inf'))
            print('epoch: %d, error: %.5f, optCond: %.5f' % (i, f, optCond))
            self.weights = utils.unflatten_weights(weights_flat, self.layer_sizes)

            # Test termination conditions
            if optCond < optTol:
              if self.verbose:
                  print("Problem solved up to optimality tolerance %.3f" % optTol)
              break

    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
        if self.classification:
            return np.argmax(Z, axis=1)
        else:
            return Z
    def score(self, X, y):
        return np.mean(self.predict(X) == y)






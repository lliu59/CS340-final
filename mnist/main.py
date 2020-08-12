# standard import
import os
import pickle
import gzip
import argparse
import numpy as np
import time

# external packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier

# internal preprocessing and utit tools
from preprocessing import deskew, augment_with_elastic_distortions
import utils

# classifiers
from classifiers.knn import KNN
from classifiers.logReg import multiLogRegL2
from classifiers.svm import multiSVML2, kernelSVML2
from classifiers.mlp import mlp
from classifiers.cnn import create_model

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    # load mnist dataset
    with gzip.open(os.path.join('./', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xtest, ytest = test_set

    # binarize data
    binarizer = LabelBinarizer()
    Y = binarizer.fit_transform(y)  # one hot encoding

    # deskew X and Xtest
    X_deskewed = deskew(X)
    Xtest_deskewed = deskew(Xtest)

    if question == "1": # KNN
        """
        KNN with PCA
        """

        # cv: find best hyperparameter combination
        pipe = Pipeline([('pca', PCA()),('knn', KNN())], verbose=1) # initialize cv pipeline
        params_grid = {
            'pca__n_components': np.arange(5, 100, 5),
            'knn__k': np.arange(2, 11)
        }
        gs = GridSearchCV(pipe, params_grid, cv=5, verbose=1, n_jobs=-1)  # use all processors
        gs.fit(X_deskewed, y)
        best_parameters = gs.best_estimator_.get_params() # get best params
        n_components = best_parameters['pca__n_components']
        k = best_parameters['knn__k']

        # run PCA on optimal n_components value
        pca = PCA(n_components=n_components, svd_solver='full')
        X_pca = pca.fit_transform(X_deskewed)
        Xtest_pca = pca.transform(Xtest_deskewed)

        # training
        model = KNN(k=k)
        model.fit(X_pca, y)

        # prediction
        y_train_pred = model.predict(X_pca)
        y_pred = model.predict(Xtest_pca)
        print("Training error %.5f" % np.mean(y_train_pred != y))
        print("Test error %.5f" % np.mean(y_pred != ytest))

    elif question == '2': # logistic regression
        """
        multi-class logistic regression with PCA, softmax loss and L2-regularization
        """

        # cv: find best hyperparameter combination
        pipe = Pipeline([('pca', PCA()), ('logReg', multiLogRegL2())], verbose=1)  # initialize cv pipeline
        params_grid = {
            'pca__n_components': np.arange(5, 100, 5),
            'logReg__lam': [0.001, 0.1, 1, 100]
        }
        gs = GridSearchCV(pipe, params_grid, cv=5, verbose=1, n_jobs=-1)  # use all processors
        gs.fit(X_deskewed, y)
        best_parameters = gs.best_estimator_.get_params()  # get best params
        n_components = best_parameters['pca__n_components']
        lam = best_parameters['logReg__lam']

        # run PCA on optimal n_components value
        pca = PCA(n_components=n_components, svd_solver='full')
        X_pca = pca.fit_transform(X_deskewed)
        Xtest_pca = pca.transform(Xtest_deskewed)

        # training
        model = multiLogRegL2(lam=lam, maxEvals=500)
        model.fit(X_pca, y)

        # prediction
        y_train_pred = model.predict(X_pca)
        y_pred = model.predict(Xtest_pca)
        print("Training error %.5f" % np.mean(y_train_pred != y))
        print("Test error %.5f" % np.mean(y_pred != ytest))

    elif question == '3': # SVM
        """
        Attempt 2 (adpoted): multi-class SVM with polynomial kernel, PCA and L2-regularization
        - trained on a subsample of training set (size = 25,000) 
        - best test error: 2.126%
        """
        # cv: find best hyperparameter combination
        pipe = Pipeline([('pca', PCA()), ('svm', kernelSVML2())], verbose=1)  # initialize cv pipeline
        params_grid = {
            'pca__n_components': np.arange(5, 100, 5),
            'svm__lam': [0.001, 0.1, 1, 100],
            'svm__p': np.arange(2, 11),
            'svm__coef': [0.01, 0.1, 1, 10]
        }
        gs = GridSearchCV(pipe, params_grid, cv=5, verbose=1, n_jobs=-1)  # use all processors
        gs.fit(X_deskewed, y)
        best_parameters = gs.best_estimator_.get_params()  # get best params
        n_components = best_parameters['pca__n_components']
        lam = best_parameters['svm__lam']
        p = best_parameters['svm__p']
        coef = best_parameters['svm__coef']

        # run PCA on optimal n_components value
        pca = PCA(n_components=n_components, svd_solver='full')
        X_pca = pca.fit_transform(X_deskewed)
        Xtest_pca = pca.transform(Xtest_deskewed)

        # training (trained with half of training to avoid long runtime)
        sub = 25000
        rand_ind = np.random.choice(50000, sub)
        model = kernelSVML2(p=p, coef=coef, lam=lam, maxEvals=500)
        model.fit(X_pca[rand_ind], y[rand_ind])

        # prediction
        y_train_pred = model.predict(X_pca)
        y_pred = model.predict(Xtest_pca)
        print("Training error %.5f" % np.mean(y_train_pred != y))
        print("Test error %.5f" % np.mean(y_pred != ytest))

        """
        Attempt 1 (discarded): multi-class linear SVM (non-kernelized) with PCA and L2-regularization
        - best test error = 4.920%
        """

        # cv: find best hyperparameter combination
        pipe = Pipeline([('pca', PCA()), ('svm', multiSVML2())], verbose=1)  # initialize cv pipeline
        params_grid = {
            'pca__n_components': np.arange(5, 100, 5),
            'svm__lam': [0.001, 0.1, 1, 100]
        }
        gs = GridSearchCV(pipe, params_grid, cv=5, verbose=1, n_jobs=-1)  # use all processors
        gs.fit(X_deskewed, y)
        best_parameters = gs.best_estimator_.get_params()  # get best params
        n_components = best_parameters['pca__n_components']
        lam = best_parameters['svm__lam']

        # run PCA on optimal n_components value
        pca = PCA(n_components=n_components, svd_solver='full')
        X_pca = pca.fit_transform(X_deskewed)
        Xtest_pca = pca.transform(Xtest_deskewed)

        # training
        model = multiSVML2(lam=lam, maxEvals=500)
        model.fit(X_pca, y)

        # prediction
        y_train_pred = model.predict(X_pca)
        y_pred = model.predict(Xtest_pca)
        print("Training error %.5f" % np.mean(y_train_pred != y))
        print("Test error %.5f" % np.mean(y_pred != ytest))

    elif question == '4': # MLP
        """
        MLP with L2-regularization and GD
        """

        """
        alternative preprocessing scheme using elastic distortion: 
        commented out due to less ideal performance
        """
        # X_elas, y_elas = augment_with_elastic_distortions(X, y, alpha_range=[8, 10], sigma=3, width_shift_range=2,
        #                                                   height_shift_range=2, zoom_range=0, iterations=100)
        # Y_elas = binarizer.fit_transform(y_elas)  # one hot encoding
        # X_deskewed = deskew(X_elas)
        # Xtest_deskewed = deskew(Xtest)

        # cv: find best hyperparameter combination
        params_grid = {
            'hidden_layer_sizes': [[128], [256], [512]],
            'lammy': [0.1, 1.0],
            'alpha': [0.0001, 0.1, 1.0],
            'max_iter': [500]
        }
        gs = GridSearchCV(mlp(), params_grid, cv=3, verbose=10, n_jobs=-1)  # use all processors
        gs_result = gs.fit(X_deskewed, Y)
        print("Best: %f using %s" % (gs_result.best_score_, gs_result.best_params_))

        # compute test error
        yhat = gs.predict(Xtest_deskewed)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    elif question == '5': # CNN
        # one-hot encode Ytest for keras
        Ytest = binarizer.fit_transform(ytest)

        # convert X and Xtest to 4-dim
        X_4dim = X_deskewed.reshape(len(X_deskewed), 28, 28, 1)
        Xtest_4dim = Xtest_deskewed.reshape(len(Xtest_deskewed), 28, 28, 1)
        num_classes = np.unique(y).size
        input_shape = (28, 28, 1)

        # cv: find best hyperparameter combination
        params_grid = {
            'batch_size': [16, 64, 128, 512],
            'epochs': [8, 10, 12],
            'learn_rate': [0.001, 0.01, 0.1, 0.2],
            'fc_neurons': [64, 128],
            'conv1_neurons': [32, 64],
            'conv2_neurons': [64, 128]
        }
        model = KerasClassifier(build_fn=create_model, input_shape=input_shape, num_classes=num_classes)
        gs = GridSearchCV(model, params_grid, cv=3, verbose=10, n_jobs=-1)
        gs_result = gs.fit(X_4dim, Y, verbose=1,
                           validation_data=(Xtest_4dim, Ytest))  # to monitor progress only, not used for training
        print("Training error: %f using %s" % (gs_result.best_score_, gs_result.best_params_))

        # compute test error
        yhat = gs.predict(Xtest_4dim)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    else:
        print("Unknown question: %s" % question)    
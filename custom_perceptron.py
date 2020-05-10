from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

class myPerceptron(BaseEstimator, ClassifierMixin):  
    
    """This is a handmade perceptron algorithm with hyperparameter selection 
    of learning rate and number of epochs. It is built onto the scikit learn
    estimator base which renders it capable of being used almost as a scikit 
    classifier would be. The greatest advantage of sych an implementation is
    that we can apply GridSearchCV on the model as if it was a scikit-learn
    built-in classifier"""

    def __init__(self, n_epoch = 1000, l_rate = 0.001):
        self.n_epoch = n_epoch
        self.l_rate = l_rate
    
    def get_params(self, deep=True):
        return {"n_epoch": self.n_epoch, "l_rate": self.l_rate}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    # Make a prediction with weights
    def predict_row(self, row, weights):
        activation = weights[0]
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0

    # Estimate Perceptron weights using stochastic gradient descent
    def fit(self, X_train, y_train):
        train = np.hstack([np.array(X_train), np.array(y_train).reshape(-1,1)])
        n_epoch = self.n_epoch
        l_rate = self.l_rate
        print(len)
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            for row in train:
                prediction = self.predict_row(row, weights)
                error = row[-1] - prediction
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        self.weights = weights
        return weights
        raise NotImplementedError 
        

    # predict function: analogous to scikit
    def predict(self, X_test):
        weights = self.weights
        y_pred=[]
        for row in np.hstack((X_test, np.zeros(X_test.shape[0]).reshape(-1,1))):
            prediction = self.predict_row(row, weights)
            y_pred.append(prediction)
        return np.array(y_pred)
            
    def score(self, X, y):
        y_pred = self.predict(X)
        correct = 0
        for k in range(len(X)):
            if int(y_pred[k]) == int(y[k]):
                correct += 1     
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        return (correct / len(X))
        raise NotImplementedError 
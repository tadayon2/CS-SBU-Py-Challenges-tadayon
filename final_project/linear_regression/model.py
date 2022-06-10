import numpy as np

class LinearRegression:

    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, X, Y):

        # m = training no. , n = features no.
        self.m , self.n = X.shape

        # Parameters Initialization
        self.W = np.zeros(self.n)
    
        # Bias
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        #  Gradient Descent
        Y_pred = self.predict(self.X)

        #  Calculate Gradients 
        dW = - ( 2 * (self.X.T).dot(self.Y - Y_pred) ) / self.m
        db = - ( 2 * np.sum(self.Y - Y_pred) ) / self.m
        
        #  Update W
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        
        return self
    
    def predict(self, X):
        return X.dot(self.W) + self.b
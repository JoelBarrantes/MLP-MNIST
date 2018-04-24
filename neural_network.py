import numpy as np
import math

class NeuralNetwork:

    def __init__(self,i_q, h1_q, h2_q, o_q, alpha, train_data, labels, test_data, t_labels, layers):

        self.two_layers = layers
        self.num_inputs = i_q
        self.num_hidden1 = h1_q
        self.num_hidden2 = h2_q
        self.num_outputs = o_q

        self.alpha = alpha

        self.W1 = np.random.uniform(-1, 1,(self.num_inputs, self.num_hidden1))

        self.W2 = np.random.uniform(-1, 1,(self.num_hidden1, self.num_hidden2))
        if self.two_layers:
            self.W3 = np.random.uniform(-1, 1,(self.num_hidden2,self.num_outputs))
        else:
            self.W3 = np.random.uniform(-1, 1,(self.num_hidden1,self.num_outputs))

        self.X = train_data.transpose()
        self.X = self.X / np.max(self.X)
        one_hot = np.zeros((labels.size, self.num_outputs))
        one_hot[np.arange(labels.size),labels.astype(int)] = 1
        self.Y = one_hot.copy()
        self.labels = labels

        self.X_t = test_data.transpose()
        self.X_t = self.X_t / np.max(self.X_t)
        one_hot = np.zeros((t_labels.size, self.num_outputs))
        one_hot[np.arange(t_labels.size),t_labels.astype(int)] = 1
        self.Y_t = one_hot.copy()
        self.t_labels = t_labels

    def softmax_grad(self, softmax):
        s = softmax.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)

    def softmax(self, x):
        exp_scores = np.exp(x-np.max(x))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def ReLU(self, x):
        f_x = np.maximum(0, x)
        d_x = f_x.copy()
        d_x[d_x>0] = 1
        return f_x, d_x

    def cross_entropy(self, f_w3):
        loss = 1/f_w3.shape[0] * -np.sum((self.Y * np.log(f_w3)))
        return loss

    def calculate_forward(self, X):

        w1 = np.dot(X, self.W1)
        f_w1, d_w1 = self.ReLU(w1)

        if self.two_layers:
            w2 = np.dot(f_w1, self.W2)
            f_w2, d_w2 = self.ReLU(w2)
        else:
            f_w2=f_w1
            d_w2=d_w1
        w3 = np.dot(f_w2, self.W3)
        f_w3 = self.softmax(w3)

        return f_w1, f_w2, f_w3, d_w1, d_w2

    def calculate_backward(self, f_w1, f_w2, f_w3, d_w1, d_w2):

        #CROSS ENTROPY DERIVATIVE
        deltaz3 = f_w3 - self.Y
        size = self.X.shape[0]

        if self.two_layers:
            deltaf_w3 = np.dot(deltaz3, self.W3.T)
            deltaz2 = deltaf_w3 * d_w2
            deltaf_w2 = np.dot(deltaz2, self.W2.T)
            deltaz1 = deltaf_w2 * d_w1
            grad_w1 = np.dot(self.X.T, deltaz1)/ size
            grad_w2 = np.dot(f_w1.T, deltaz2)/ size
            grad_w3 = np.dot(f_w2.T, deltaz3)/ size


        else:
            deltaf_w3 = np.dot(deltaz3, self.W3.T)
            deltaz2 = deltaf_w3 * d_w2
            grad_w1 = np.dot(self.X.T, deltaz2)/ size
            grad_w2 = grad_w1
            grad_w3 = np.dot(f_w2.T, deltaz3)/ size



        return grad_w1, grad_w2, grad_w3

    def calculate_error(self, f_w3):
        return self.cross_entropy(f_w3)

    def calculate_accuracy(self):
        f_w1, f_w2, f_w3, d_w1, d_w2 = self.calculate_forward(self.X_t)
        t = (f_w3 == f_w3.max(axis=1, keepdims=1)).astype(float)
        return np.sum(np.equal(t,self.Y_t).all(axis=1))/self.X_t.shape[0]

    def update_weights(self, grad_w1, grad_w2, grad_w3):

        self.W1 = self.W1 - self.alpha * grad_w1

        if self.two_layers:
            self.W2 = self.W2 - self.alpha * grad_w2

        self.W3 = self.W3 - self.alpha * grad_w3

    def train(self, epochs):

        for i in range(0, epochs):

            f_w1, f_w2, f_w3, d_w1, d_w2 = self.calculate_forward(self.X)

            loss = self.calculate_error(f_w3)
            acc = self.calculate_accuracy()
            print(loss)
            print(acc)

            grad_w1, grad_w2, grad_w3 = self.calculate_backward(f_w1,f_w2, f_w3, d_w1, d_w2)

            self.update_weights(grad_w1, grad_w2, grad_w3)


import numpy as np
import math
import time

class NeuralNetwork:

    def __init__(self,i_q, h1_q, h2_q, o_q, alpha, layers):

        self.two_layers = layers
        self.num_inputs = i_q
        self.num_hidden1 = h1_q
        self.num_hidden2 = h2_q
        self.num_outputs = o_q

        self.alpha = alpha

        mu = 0
        sigma = 0.01

        self.W1 = np.random.normal(mu, sigma,(self.num_inputs, self.num_hidden1))
        if self.two_layers:
            self.W2 = np.random.normal(mu, sigma,(self.num_hidden1, self.num_hidden2))
            self.W3 = np.random.normal(mu, sigma,(self.num_hidden2,self.num_outputs))
        else:
            self.W2 = np.zeros(1)
            self.W3 = np.random.normal(mu, sigma,(self.num_hidden1,self.num_outputs))

        self.date = time.strftime("%Y.%m.%d-%H.%M.%S")

    def load_data(self, train_data, labels, test_data, t_labels,):

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

    def load_Ws(self, w1, w2, w3):
        self.W1 = w1
        self.W2 = w2
        self.W3 = w3

    def softmax_single(self, sample):
        exp_scores = np.exp(sample-np.max(sample))
        return exp_scores/np.sum(exp_scores)

    def softmax(self, x):
        exp_scores = np.exp(x-np.max(x))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def ReLU(self, x):
        f_x = np.maximum(0, x)
        d_x = f_x.copy()
        d_x[d_x>0] = 1
        return f_x, d_x

    def saveWs(self,W1,W2,W3):
        date = self.date
        np.save("./File/W1_" + str(self.num_inputs) + "_" + str(self.num_hidden1) + "_" + date, W1)
        if self.two_layers:
            np.save("./File/W2_" + str(self.num_hidden1) + "_" + str(self.num_hidden2) + "_" + date, W2)
            np.save("./File/W3_" + str(self.num_hidden2) + "_" + str(self.num_outputs) + "_" + date, W3)
        else:
            np.save("./File/W3_" + str(self.num_hidden1) + "_" + str(self.num_outputs) + "_" + date, W3)

    def cross_entropy(self, f_w3, Y):
        loss = 1/f_w3.shape[0] * -np.sum((Y * np.log(f_w3)))
        return loss

    def calculate_forward(self, X, p_h):

        w1 = np.dot(X, self.W1)
        f_w1, d_w1 = self.ReLU(w1)

        #DROPOUT HIDDEN LAYER 1
        mask_h1 = np.random.binomial(size=f_w1.shape[1], n=1, p=p_h)/p_h
        f_w1 = f_w1 * mask_h1

        if self.two_layers:
            w2 = np.dot(f_w1, self.W2)
            f_w2, d_w2 = self.ReLU(w2)

            #DROPOUT HIDDEN LAYER 2
            mask_h2 = np.random.binomial(size=f_w2.shape[1], n=1, p=p_h)/p_h
            f_w2 = f_w2 * mask_h2

        else:
            f_w2 = f_w1
            d_w2 = d_w1
            mask_h2 = mask_h1
        w3 = np.dot(f_w2, self.W3)
        f_w3 = self.softmax(w3)

        return f_w1, f_w2, f_w3, d_w1, d_w2, mask_h1, mask_h2

    def calculate_backward(self, f_w1, f_w2, f_w3, d_w1, d_w2, X, Y, mh1, mh2):

        #CROSS ENTROPY DERIVATIVE
        deltaz3 = f_w3 - Y
        deltaf_w3 = np.dot(deltaz3, self.W3.T)

        size = X.shape[0]

        if self.two_layers:

            d_w2 = d_w2 * mh2
            deltaz2 = deltaf_w3 * d_w2
            deltaf_w2 = np.dot(deltaz2, self.W2.T)

            d_w1 = d_w1 * mh1
            deltaz1 = deltaf_w2 * d_w1
            grad_w1 = np.dot(X.T, deltaz1)/ size
            grad_w2 = np.dot(f_w1.T, deltaz2)/ size
            grad_w3 = np.dot(f_w2.T, deltaz3)/ size


        else:

            d_w1 = d_w1 * mh1
            deltaz1 = deltaf_w3 * d_w1

            grad_w1 = np.dot(X.T, deltaz1)/ size
            grad_w2 = grad_w1
            grad_w3 = np.dot(f_w2.T, deltaz3)/ size

        return grad_w1, grad_w2, grad_w3

    def calculate_error(self, f_w3, Y):
        return self.cross_entropy(f_w3, Y)

    def calculate_accuracy_test(self, X, Y):
        f_w1, f_w2, f_w3, d_w1, d_w2, mh1, mh2 = self.calculate_forward(X, 1)
        t = (f_w3 == f_w3.max(axis=1, keepdims=1)).astype(float)
        return np.sum(np.equal(t,Y).all(axis=1))/X.shape[0]

    def update_weights(self, grad_w1, grad_w2, grad_w3):

        self.W1 = self.W1 - self.alpha * grad_w1

        if self.two_layers:
            self.W2 = self.W2 - self.alpha * grad_w2

        self.W3 = self.W3 - self.alpha * grad_w3

    def train_batch(self, epochs, batch_size, p_h):

        idx = np.arange(self.X.shape[0])
        for j in range(0, epochs):
            start = time.time()
            np.random.shuffle(idx)
            i=0
            Loss = 0
            num_batches = 0
            exit_cond = False
            while True:
                if i+batch_size >= self.X.shape[0]:
                    exit_cond = True
                    batch = self.X[idx[i:self.X.shape[0]]]
                    batch_l = self.Y[idx[i:self.Y.shape[0]]]
                else:
                    batch = self.X[idx[i:i+batch_size]]
                    batch_l = self.Y[idx[i:i+batch_size]]

                f_w1, f_w2, f_w3, d_w1, d_w2, mh1, mh2 = self.calculate_forward(batch, p_h)
                loss = self.calculate_error(f_w3, batch_l)
                Loss += loss
                grad_w1, grad_w2, grad_w3 = self.calculate_backward(f_w1,f_w2, f_w3, d_w1, d_w2, batch, batch_l, mh1, mh2)

                self.update_weights(grad_w1, grad_w2, grad_w3)
                i += batch_size
                num_batches += 1
                batch=None
                batch_l=None
                if exit_cond:
                    break

            self.saveWs(self.W1, self.W2, self.W3)
            acc = self.calculate_accuracy_test(self.X_t, self.Y_t)
            end = time.time()


            print("Epoch: ", j)
            print("Loss", Loss/num_batches)
            print("Accuracy: ", acc)
            print("Duration: ", end - start)


    def train_all(self, epochs, p_h):

        for j in range(0, epochs):
            start = time.time()
            f_w1, f_w2, f_w3, d_w1, d_w2, mh1, mh2 = self.calculate_forward(self.X, p_h)

            loss = self.calculate_error(f_w3, self.Y)
            acc = self.calculate_accuracy_test(self.X_t, self.Y_t)

            grad_w1, grad_w2, grad_w3 = self.calculate_backward(f_w1,f_w2, f_w3, d_w1, d_w2, self.X, self.Y, mh1, mh2)

            self.update_weights(grad_w1, grad_w2, grad_w3)

            self.saveWs(self.W1, self.W2, self.W3)

            end = time.time()
            print("Epoch: ", j)
            print("Loss", loss)
            print("Accuracy: ", acc)
            print("Duration: ", end - start)


    def test_image(self, sample):
        sample = sample/sample.max()
        w1 = np.dot(sample, self.W1)
        f_w1, _ = self.ReLU(w1)
        if self.two_layers:
            w2 = np.dot(f_w1, self.W2)
            f_w2, _ = self.ReLU(w2)
        else:
            f_w2 = f_w1
        w3 = np.dot(f_w2, self.W3)
        f_w3 = self.softmax_single(w3)
        return np.argmax(f_w3)



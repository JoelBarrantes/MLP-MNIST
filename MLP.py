import numpy as np
from neural_network import NeuralNetwork

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

#0 ReLU
#1 Sigmoid
f_activation = 0

dropout_p = 0.5
two_layers = True
red = NeuralNetwork(784, 32, 16, 10, 0.0085, two_layers, f_activation )


red.load_data(train_img.T, train_lbl.T,test_img.T, test_lbl)
#red.train_all(10000, dropout_p)cdf

#LOAD PREVIOUS WEIGHTS
w1 = np.load("./File/W1_784_32_2018.05.09-20.24.51.npy")
w2 = np.load("./File/W2_32_16_2018.05.09-20.24.51.npy")
w3 = np.load("./File/W3_16_10_2018.05.09-20.24.51.npy")

#red.load_Ws(w1,w2,w3)
#red.no_log()

red.train_batch(500, 32, dropout_p)

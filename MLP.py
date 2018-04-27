import numpy as np
from neural_network import NeuralNetwork

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)


dropout_p = 0.5
two_layers = False
red = NeuralNetwork(784, 100, 10, 10, 0.0085, train_img.T, train_lbl.T,test_img.T, test_lbl,two_layers )

red.train_all(10000, dropout_p)

#red.train_batch(10000, 32, dropout_h)

print("Here")

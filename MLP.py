import numpy as np
from neural_network import NeuralNetwork

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)


dropout_p = 0.5
two_layers = True
red = NeuralNetwork(784, 50, 30, 10, 0.0085, two_layers )
red.load_data(train_img.T, train_lbl.T,test_img.T, test_lbl)
#red.train_all(10000, dropout_p)cdf

#LOAD PREVIOUS WEIGHTS
w1 = np.load("./File/W1_784_50_2018.04.27-18.02.27.npy")
w2 = np.load("./File/W2_50_30_2018.04.27-18.02.27.npy")
w3 = np.load("./File/W3_30_10_2018.04.27-18.02.27.npy")


#red.load_Ws(w1,w2,w3)

red.train_batch(10000, 32, dropout_p)

print("Here")

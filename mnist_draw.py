import numpy as np
import scipy.misc as misc


def save_weights(imagen,i):
    misc.imsave('./images/neuron '+i+".png", imagen)

w1 = np.load("./File/W1_784_100_2018.05.03-14.06.23.npy")
w1 = w1.T
i = 1
for imagen in w1:
    save_weights(np.reshape(imagen,(28,28)), str(i))
    i+=1


import sys
import os
import numpy as np
from neural_network import softmax
from neural_network_2 import cross_entropy_error
from numerical import numerical_gradient

sys.path.append(os.pardir)


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p)

t = np.array([0, 0, 1])
net.loss(x, t)

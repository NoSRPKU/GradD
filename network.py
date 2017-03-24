# -*- coding: utf-8 -*-

import numpy as np
import scipy
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return np.array([(1. - math.exp(-a))/(1. + math.exp(-a)) for a in x])

def sigmoid_rev(x):
    return np.array([-math.log((1. - a) / (a + 1.)) for a in x])

class Network(object):

    def __init__(self, node_amounts, activation_fn, rev_fn):
        self.node_amounts = node_amounts
        self.activation_fn = activation_fn
        self.rev_fn = rev_fn
        #random bias&weight in [0, 1)
        self.bias = []
        self.weight = []
        for i, j in zip(node_amounts[:-1], node_amounts[1:]):
            _weight = np.random.random(size = i * j).reshape(i, j) * 6 - 3
            _bias = np.random.random(size=j)
            self.weight.append(_weight)
            self.bias.append(_bias)

    def compute(self, inp):
        if len(inp) != self.node_amounts[0]:
            return np.zeros(self.node_amounts[-1])
        layer_output = np.array(inp)
        for _weight, _bias in zip(self.weight, self.bias):
            #print "weight", _weight
            #print "bias", _bias
            layer_output = np.dot(layer_output, _weight)
            #print "lop", layer_output
            layer_output += _bias
            #print "lop2", layer_output
            layer_output = self.activation_fn(layer_output)
            print "lopsg", layer_output
        return layer_output

    def reconstruct(self, outp):
        layer_reverse = np.array(outp)
        print "lr", layer_reverse
        for _weight, _bias in zip(self.weight, self.bias)[::-1]:
            layer_reverse = self.rev_fn(layer_reverse)
            #print "lrsg-0", layer_reverse
            layer_reverse -= _bias
            #print "lrsg-1", layer_reverse
            u, sigma, vt = np.linalg.svd(_weight)
            sig = np.zeros((len(vt), len(u)))
            for i in range(0, len(sigma)):
                sig[i][i] = 1./sigma[i] if math.fabs(sigma[i]) > 1e-10 else 0
            rev = np.dot(vt.transpose(), sig)
            rev = np.dot(rev, u.transpose())
            layer_reverse = np.dot(layer_reverse, rev)
            print "lrsvd", layer_reverse
        return layer_reverse

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0):
        pass

if __name__ == "__main__":
    network = Network([2, 3, 5, 5, 6], sigmoid, sigmoid_rev)
    #print network.bias
    #print network.weight
    v = network.compute([2, 3])
    print network.reconstruct(v)
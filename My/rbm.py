# -*- coding: utf-8 -*-

import theano
import numpy as np
import scipy
import theano.tensor as T

from const import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class RBM(object):

    def __init__(self,
                 n_visible = 1,
                 n_hidden = 1,
                 W = None,
                 bias_visible = None,
                 bias_hidden = None,
                 numpy_random_number_generator = None,
                 theano_random_number_generator = None,
                 sample_val = None
                 ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        uniform_number = 4. * np.sqrt(6. / (n_hidden + n_visible))

        if not numpy_random_number_generator:
            numpy_random_number_generator = np.random.RandomState(numpy_random_generator_seed)
        if not theano_random_number_generator:
            theano_random_number_generator = RandomStreams(numpy_random_number_generator.randint(2 ** 30))
        self.theano_random_number_generator = theano_random_number_generator

        if not W:
            W = theano.shared(value=np.asarray(
                numpy_random_number_generator.uniform(low=-uniform_number, high=uniform_number, size=(n_visible, n_hidden)),
                dtype=theano.config.floatX
            ), name='W', borrow=True)
        self.W = W

        if not bias_hidden:
            bias_hidden = theano.shared(value=np.zeros(
                n_hidden, dtype=theano.config.floatX
            ), name='hbias', borrow=True)
        self.bias_hidden = bias_hidden

        if not bias_visible:
            bias_visible = theano.shared(value=np.zeros(
                n_visible, dtype=theano.config.floatX
            ), name='vbias', borrow=True)
        self.bias_visible = bias_visible

    def sample_h_given_v(self, v):
        pre_sigmoid_h = T.dot(v, self.W) + self.bias_hidden
        h = T.nnet.sigmoid(pre_sigmoid_h)
        sample_h = self.theano_random_number_generator.binomial(size=h.shape, p=h, dtype=theano.config.floatX)
        return pre_sigmoid_h, h, sample_h

    def sample_v_given_h(self, h):
        pre_sigmoid_v = T.dot(v, self.W.T) + self.bias_visible
        v = T.nnet.sigmoid(pre_sigmoid_v)
        sample_v = self.theano_random_number_generator.binomial(size=v.shape, p=v, dtype=theano.config.floatX)
        return pre_sigmoid_v, v, sample_v

    def train(self, sample, epoch, learning_rate, k):
        dW, dbias_visible, dbias_hidden = self.cdk(sample, k)

    def cdk(self, sample, k):
        for samp in sample:
            v = samp
            for r in range(0, k):
                pass
            pass

if __name__ == "__main__":
    rbm = RBM()
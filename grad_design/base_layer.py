# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import time
import numpy as np

DTYPE = theano.config.floatX

class BaseLayer(object):

    def __init__(
            self,
            n_visible, n_hidden,
            bias_visible=None, bias_hidden=None,
            W=None, WT = None, init_w_limit=None,
            numpy_rng_seed=None, theano_rng_seed=None,
            activation=None,
            **kwargs
    ):

        if not n_visible or not n_hidden:
            raise AttributeError("BaseLayer init n_visible and n_hidden cannot be zero")
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng_seed is None:
            numpy_rng = np.random.RandomState(int(time.time()))
        else:
            numpy_rng = np.random.RandomState(numpy_rng_seed)
        self.numpy_rng = numpy_rng

        if theano_rng_seed is None:
            theano_rng_seed = numpy_rng.randint(2 ** 30)
        self.theano_rng = RandomStreams(theano_rng_seed)

        if W is None:
            if init_w_limit is None:
                init_w_limit = 4. * np.sqrt(6. / (n_visible + n_hidden))
            self.W = theano.shared(
                value=np.asarray(
                    numpy_rng.uniform(
                        low=-init_w_limit,
                        high=init_w_limit,
                        size=(n_visible, n_hidden)
                    ),
                    dtype=DTYPE
                ),
                name='W'
            )
        elif isinstance(W, list):
            init_W = np.asarray(W, dtype=DTYPE)
            if init_W.shape != (n_visible, n_hidden):
                raise AttributeError("BaseLayer init shape of W is illegal: %r, n_visible=%r, n_hidden=%r" %
                                          (init_W.shape, n_visible, n_hidden))
            self.W = theano.shared(
                value=init_W,
                name='W',
                borrow=True
            )
        elif isinstance(W, np.ndarray):
            shape = W.shape
            if shape != (n_visible, n_hidden):
                raise AttributeError("BaseLayer init shape of W is illegal: %r, n_visible=%r, n_hidden=%r" %
                                          (shape, n_visible, n_hidden))
            self.W = theano.shared(W, name='W')
        elif isinstance(W, theano.Variable):
            self.W = W
        else:
            raise AttributeError("BaseLayer init type of W is illegal: %r" % type(W))

        if WT is None:
            if init_w_limit is None:
                init_w_limit = 4. * np.sqrt(6. / (n_visible + n_hidden))
            self.WT = theano.shared(
                value=np.asarray(
                    numpy_rng.uniform(
                        low=-init_w_limit,
                        high=init_w_limit,
                        size=(n_hidden, n_visible)
                    ),
                    dtype=DTYPE
                ),
                name='WT'
            )
        elif isinstance(WT, list):
            init_WT = np.asarray(WT, dtype=DTYPE)
            if init_WT.shape != (n_hidden, n_visible):
                raise AttributeError("BaseLayer init shape of WT is illegal: %r, n_visible=%r, n_hidden=%r" %
                                          (init_WT.shape, n_hidden, n_visible))
            self.WT = theano.shared(
                value=init_WT,
                name='WT',
                borrow=True
            )
        elif isinstance(WT, np.ndarray):
            shape = WT.shape
            if shape != (n_visible, n_hidden):
                raise AttributeError("BaseLayer init shape of WT is illegal: %r, n_visible=%r, n_hidden=%r" %
                                          (shape, n_visible, n_hidden))
            self.WT = theano.shared(WT, name='WT')
        elif isinstance(WT, theano.Variable):
            self.WT = WT
        else:
            raise AttributeError("BaseLayer init type of WT is illegal: %r" % type(WT))

        if bias_visible is None:
            self.bias_visible = theano.shared(
                value=np.zeros(shape=n_visible, dtype=DTYPE),
                name='b_vis',
                borrow=True
            )
        elif isinstance(bias_visible, list):
            init_bias_visible = np.asarray(bias_visible, dtype=DTYPE)
            if init_bias_visible.shape != (n_visible,):
                raise AttributeError("BaseLayer init shape of bias_visible is illegal: %r, n_visible=%r" %
                                          (init_bias_visible.shape, n_visible))
            self.bias_visible = theano.shared(value=init_bias_visible, name='b_vis')
        elif isinstance(bias_visible, theano.Variable):
            self.bias_visible = bias_visible
        else:
            raise AttributeError("BaseLayer init type of bias_visible is illegal: %r" % type(bias_visible))

        if bias_hidden is None:
            self.bias_hidden = theano.shared(
                value=np.zeros(shape=n_hidden, dtype=DTYPE),
                name='b_hid',
                borrow=True
            )
        elif isinstance(bias_hidden, list):
            init_bias_hidden = np.asarray(bias_hidden, dtype=DTYPE)
            if init_bias_hidden.shape != (n_hidden,):
                raise AttributeError("BaseLayer init shape of bias_hidden is illegal: %r, n_hidden=%r" %
                                          (init_bias_hidden.shape, n_hidden))
            self.bias_hidden = theano.shared(value=init_bias_hidden, name='b_hid')
        elif isinstance(bias_hidden, theano.Variable):
            self.bias_hidden = bias_hidden
        else:
            raise AttributeError("BaseLayer init type of bias_hidden is illegal: %r" % type(bias_hidden))

        if activation is None:
            activation = T.nnet.sigmoid
        self.activation = activation

        self.params = [self.W, self.bias_visible, self.bias_hidden]

    def count_hidden_by_visible(self, v, *args, **kwargs):
        return self.activation(T.dot(v, self.W) + self.bias_hidden)

    def get_cost(self, *args, **kwargs):
        raise NotImplementedError("BaseLayer get_cost isn't implemented in class BaseLayer")

    def get_cost_gparams(self, *args, **kwargs):
        cost = self.get_cost(*args, **kwargs)
        gparams = T.grad(cost, self.params)
        return cost, gparams

    def get_updates(self, gparams, learnig_rate, *args, **kwagrs):
        updates = [(param, param - learnig_rate * gparam) for param, gparam in zip(self.params, gparams)]
        return updates

if __name__ == "__main__":
    base_layer = BaseLayer(2, 1, W=[[2], [1]])
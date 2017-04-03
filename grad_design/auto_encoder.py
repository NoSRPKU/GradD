# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np
from base_layer import BaseLayer, DTYPE
from loaddata import load_data
import time

import csv

class AutoEncoder(BaseLayer):

    def __init__(self, *args, **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.params = [self.W, self.WT, self.bias_hidden, self.bias_visible]
        self.pre_cost = None
        self.out_cost = {}

    def get_corrupted_input(self, input_val, corrupt):
        return input_val * self.theano_rng.binomial(size=input_val.shape, p=1 - corrupt, dtype=DTYPE)

    def get_reconstructed_visible(self, hidden):
        return self.activation(T.dot(hidden, self.WT) + self.bias_visible)

    def get_cost(self, input_val=None, corrupt=0, need_activation=True, *args, **kwargs):
        if input_val is None:
            raise AttributeError("AutoEncoder get_cost inpu_val is None")
        if isinstance(input_val, list):
            input_val = np.asarray(input_val, dtype=DTYPE)
        if need_activation:
            input_val = self.activation(input_val)
        corrupted_v = self.get_corrupted_input(input_val=input_val, corrupt=corrupt)
        h = self.count_hidden_by_visible(corrupted_v)
        reconstructed_v = self.get_reconstructed_visible(h)

        L = -T.sum(input_val * T.log(reconstructed_v) + (1 - input_val) * T.log(1 - reconstructed_v), axis=1)
        cost = T.mean(L)
        return cost

    def get_batch_cost_updates(self, input_val, begin=0, end=-1, corrupt=0):
        batch_input_val = input_val.get_value(borrow=True)[begin:end]
        print batch_input_val

    def get_training_batch_function(self, input_v, learning_rate=0.1):
        input_val = T.matrix('v', dtype=DTYPE)
        begin = T.lscalar()
        end = T.lscalar()

        cost, gparams = self.get_cost_gparams(input_val=input_val, need_activation=True)
        updates = self.get_updates(gparams=gparams, learnig_rate=learning_rate)

        training_batch = theano.function(
            inputs=[begin, end],
            outputs=cost,
            updates=updates,
            givens={input_val:input_v[begin:end]}
        )
        return training_batch

    def train(self, input_v, input_size, learning_rate=0.1, training_epoch=15, batch_size=1, *args, **kvargs):
        if isinstance(input_v, list):
            input_v = np.asarray(input_v, dtype=DTYPE)
        if isinstance(input_v, np.ndarray):
            input_v = theano.shared(
                value=input_v,
                name='input_v',
                borrow=True
            )
        train_batch = self.get_training_batch_function(input_v=input_v, learning_rate=learning_rate)

        for e in range(0, training_epoch):
            index = 0
            flag_i = False
            while index + batch_size <= input_size:
                try:
                    cost = train_batch(index, index + batch_size)
                    ocst = self.out_cost.get(index)
                    #print e, index, cost, ocst, self.pre_cost
                    if ocst is not None and np.abs(ocst - cost) < 0.000001:
                        flag_i = True
                    self.out_cost[index] = cost
                except Exception, err:
                    print err
                index += batch_size
            if flag_i:
                break
            if not self.out_cost:
                break
            out_mean = np.mean(self.out_cost.values())
            if self.pre_cost is not None and np.abs(self.pre_cost - out_mean) < 0.000001:
                break
            self.pre_cost = out_mean
            if e % 50 == 0:
                print e, self.pre_cost
                self.test(*args, **kvargs)

    def test(self, test_sample):
        def thprint(x, up=None):
            f = theano.function([], x, updates=up)
            print f()
        _smp = np.asarray(test_sample, dtype=DTYPE)
        smp = T.nnet.sigmoid(_smp)
        h = self.count_hidden_by_visible(smp)
        v_rec = self.get_reconstructed_visible(h)
        v_rec2 = T.log(v_rec) - T.log(1 - v_rec)
        cost = T.mean(T.abs_(v_rec2 - _smp), axis=1)
        thprint(cost)
        cost2 = T.mean(T.abs_(v_rec - smp), axis=1)
        thprint(cost2)
        print

    def output(self, out_file):
        with open(out_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow([self.n_visible, self.n_hidden])
            writer.writerow(self.bias_visible.get_value(borrow=True))
            writer.writerow(self.bias_hidden.get_value(borrow=True))
            W = self.W.get_value(borrow=True)
            for r in W:
                writer.writerow(r)
            WT = self.WT.get_value(borrow=True)
            for r in WT:
                writer.writerow(r)
            f.close()


if __name__ == "__main__":
    auto_encoder = AutoEncoder(32, 20)
    train_inp, train_outp, train_outp_t, \
    test_inp, test_outp, test_outp_t, \
    verify_inp, verify_outp, verify_out_t = load_data(
        file_path='/home/nosr/Documents/out/LM782.csv', n=8, stdd = 1,
        train_sample_beg='2009-01-01 00:00:00',
        test_sample_beg='2013-01-01 00:00:00',
        verify_time_beg='2014-01-01 00:00:00',
        verify_time_end='2015-01-01 00:00:00'
    )
    auto_encoder.train(input_v=train_inp, input_size=len(train_inp),
                       learning_rate=0.2, training_epoch=10000, batch_size=len(train_inp) / 48, test_sample=test_inp)

    auto_encoder.test(test_inp)
    auto_encoder.test(verify_inp)
    auto_encoder.output('data.csv')
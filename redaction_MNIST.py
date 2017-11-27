#!/usr/bin/env python

import sys
import os
import time

import pandas as pd
import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.nonlinearities import leaky_rectify, tanh, sigmoid
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, TransposedConv2DLayer, ReshapeLayer

from sklearn.model_selection import train_test_split

def load_data():
    """

    :return: y is a dictionary of {confounder_name: [categories, one_hot_representation]}
    """
    gene_counts = pd.DataFrame.from_csv('RNASeqQC.all_samples.gene.counts.txt', sep='\t')
    gene_counts = gene_counts.set_index(gene_counts.iloc[:,0]).iloc[:,1:]
    X = gene_counts.transpose().as_matrix()

    confounders = ['sample_type', 'library', 'lane']

    meta = pd.DataFrame.from_csv('RNASeqQC.all_samples.meta.txt', sep='\t')[gene_counts.columns].loc[
        confounders].transpose()

    y = {}
    for confounder in confounders:
        c_unique, c_idx = np.unique(meta[confounder], return_inverse=True)
        y_one_hot[confounder] = [c_unique, np.eye(len(c_unique))[c_idx]]

    return X, y

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

class RedactionNetwork(object):
    def __init__(self, shape, inputs, targets, encoding_size=15, batch_size=32):
        self.enc_size = encoding_size
        self._shape = shape
        self.batch_size = batch_size

        enc_layer = self._build_encoder(self._shape, input_var=inputs)
        dec_layer = self._build_decoder(enc_layer)
        discr_layer = self._build_decoder(enc_layer)

        enc_output = lasagne.layers.get_output(enc_layer)
        dec_output = lasagne.layers.get_output(dec_layer)
        discr_output = lasagne.layers.get_output(discr_layer)

        dec_err = lasagne.objectives.squared_error(dec_output, inputs).mean()
        discr_err = lasagne.objectives.squared_error(discr_output, targets).mean()

        ae_loss = dec_err - discr_err
        adv_loss = discr_err

        ae_params = lasagne.layers.get_all_params(dec_layer, trainable=True)
        num_enc_params = len(lasagne.layers.get_all_params(enc_layer, trainable=True))
        adv_params = lasagne.layers.get_all_params(discr_layer, trainable=True)[num_enc_params:]

        ae_updates = lasagne.updates.adadelta(ae_loss, ae_params)
        adv_updates = lasagne.updates.adadelta(adv_loss, adv_params)

        self.ae_train_fn = theano.function([inputs, targets], ae_loss, updates=ae_updates)
        self.adv_train_fn = theano.function([inputs, targets], adv_loss, updates=adv_updates)

        ae_accuracy = 1.0 - T.mean(T.abs_(dec_output - inputs))
        adv_accuracy = 1.0 - T.mean(T.abs_(discr_output - targets))

        self.ae_accuracy_fn = theano.function([inputs], ae_accuracy)
        self.adv_accuracy_fn = theano.function([inputs, targets], adv_accuracy)

    def _build_encoder(self, shape, input_var = None):
        input_layer = lasagne.layers.InputLayer(shape=self._shape, input_var=input_var)
        layer = lasagne.layers.Conv2DLayer(input_layer, 32, 3,
                                           stride=2,
                                           W=lasagne.init.GlorotUniform(),
                                           nonlinearity=leaky_rectify)
        layer = lasagne.layers.Conv2DLayer(layer, 32, 2,
                                           stride=1,
                                           W=lasagne.init.GlorotUniform(),
                                           nonlinearity=leaky_rectify)
        layer = lasagne.layers.Conv2DLayer(layer, 24, 3,
                                           stride=2,
                                           W=lasagne.init.GlorotUniform(),
                                           nonlinearity=leaky_rectify)
        layer = lasagne.layers.Conv2DLayer(layer, 14, 2,
                                           stride=1,
                                           W=lasagne.init.GlorotUniform(),
                                           nonlinearity=leaky_rectify)
        layer = lasagne.layers.DenseLayer(layer,
                                          num_units=32,
                                          nonlinearity=leaky_rectify)
        layer = lasagne.layers.DenseLayer(layer,
                                          num_units=32,
                                          nonlinearity=leaky_rectify)
        layer = lasagne.layers.DenseLayer(layer,
                                          num_units=self.enc_size,
                                          nonlinearity=tanh)
        return layer

    def _build_decoder(self, input_layer):
        layer = lasagne.layers.DenseLayer(input_layer,
                                          num_units=100,
                                          nonlinearity=leaky_rectify)
        layer = lasagne.layers.DenseLayer(layer,
                                          num_units=100,
                                          nonlinearity=leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(layer,
                                            shape=(self.batch_size, 2, 5, 5))
        layer = lasagne.layers.TransposedConv2DLayer(layer,
                                                     num_filters=12,
                                                     filter_size=2,
                                                     stride=2,
                                                     nonlinearity=leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(layer,
                                            shape=(self.batch_size, 12, 10, 10))
        layer = lasagne.layers.TransposedConv2DLayer(layer,
                                                     num_filters=12,
                                                     filter_size=2,
                                                     stride=2,
                                                     nonlinearity=leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(layer,
                                            shape=(self.batch_size, 12, 20, 20))
        layer = lasagne.layers.TransposedConv2DLayer(layer,
                                                     num_filters=8,
                                                     filter_size=3,
                                                     stride=1,
                                                     nonlinearity=leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(layer,
                                            shape=(self.batch_size, 8, 22, 22))
        layer = lasagne.layers.TransposedConv2DLayer(layer,
                                                     num_filters=8,
                                                     filter_size=3,
                                                     stride=1,
                                                     nonlinearity=leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(layer,
                                            shape=(self.batch_size, 8, 24, 24))
        layer = lasagne.layers.TransposedConv2DLayer(layer,
                                                     num_filters=6,
                                                     filter_size=3,
                                                     stride=1,
                                                     nonlinearity=leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(layer,
                                            shape=(self.batch_size, 6, 26, 26))
        layer = lasagne.layers.TransposedConv2DLayer(layer,
                                                     num_filters=1,
                                                     filter_size=3,
                                                     stride=1,
                                                     nonlinearity=leaky_rectify)
        layer = lasagne.layers.ReshapeLayer(layer,
                                            shape=(self.batch_size, 1, 28, 28))

        return layer

    def _build_discriminator(self, input_layer):
        layer = DenseLayer(input_layer,
                           num_units=100,
                           nonlinearity=leaky_rectify)
        layer = DenseLayer(layer,
                           num_units=100,
                           nonlinearity=leaky_rectify)
        layer = DenseLayer(layer,
                           num_units=100,
                           nonlinearity=leaky_rectify)
        layer = DenseLayer(layer,
                           num_units=100,
                           nonlinearity=leaky_rectify)
        layer = DenseLayer(layer,
                           num_units=10,
                           nonlinearity=tanh)
        return layer


if __name__ == '__main__':
    # hyperparams
    ENCODING_SIZE = 200

    # other constants
    NUM_EPOCHS = 10
    BATCH_SIZE = 32

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y['sample_type'][1], test_size=0.2)

    inputs = T.tensor2('inputs')
    targets = T.tensor2('targets')
    r = RedactionNetwork((None, X_train.shape[1]), inputs, targets, encoding_size=200)

    for epoch in xrange(NUM_EPOCHS):
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, 32, shuffle=True):
            inputs, targets = batch
            train_err += r.ae_train_fn(inputs, targets)
            train_err += r.adv_train_fn(inputs, targets)
            train_batches += 1
        print("training loss: {:.6f}".format(train_err/train_batches))
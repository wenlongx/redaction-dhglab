#!/usr/bin/env python

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.nonlinearities import leaky_rectify, rectify, tanh, sigmoid, softmax
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, TransposedConv2DLayer, ReshapeLayer

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
    def __init__(self, inputs, targets, shape=None,encoding_size=15, output_size=None, alpha=0.5):
        self.enc_size = encoding_size
        self._shape = shape
        self.input_size = np.prod(self._shape[1:])
        self.output_size = output_size
        self.alpha = alpha

        enc_layer = self._build_encoder(self._shape, input_var=inputs)
        dec_layer = self._build_decoder(enc_layer)
        discr_layer = self._build_discriminator(enc_layer)

        dec_output = lasagne.layers.get_output(dec_layer)
        discr_output = lasagne.layers.get_output(discr_layer)

        dec_err = T.mean(lasagne.objectives.squared_error(inputs, dec_output)) # Squared err
        discr_err = T.mean(T.abs_(targets - discr_output)) # MAE

        ae_loss = dec_err - 0.0*discr_err
        adv_loss = discr_err
        joint_loss = (alpha * ae_loss) + (1.0 - alpha) * (10**5) * ((1.0/self.output_size) - adv_loss)**2

        ae_params = lasagne.layers.get_all_params(dec_layer, trainable=True)
        num_enc_params = len(lasagne.layers.get_all_params(enc_layer, trainable=True))
        adv_params = lasagne.layers.get_all_params(discr_layer, trainable=True)[num_enc_params:]
        joint_params = lasagne.layers.get_all_params([dec_layer, discr_layer], trainable=True)
        self.joint_params = joint_params

        ae_updates = lasagne.updates.adadelta(ae_loss, ae_params)
        adv_updates = lasagne.updates.adadelta(adv_loss, adv_params)
        joint_updates = lasagne.updates.adadelta(joint_loss, joint_params)

        self.ae_train_fn = theano.function([inputs, targets], ae_loss, updates=ae_updates)
        self.adv_train_fn = theano.function([inputs, targets], adv_loss, updates=adv_updates)
        self.joint_train_fn = theano.function([inputs, targets], joint_loss, updates=joint_updates)

        adv_accuracy = 1.0 - T.mean(T.abs_(discr_output - targets))
        self.adv_accuracy_fn = theano.function([inputs, targets], adv_accuracy)

        self.discr_pred_fn = theano.function([inputs], discr_output)
        self.dec_pred_fn = theano.function([inputs], dec_output)

    def _build_encoder(self, shape, input_var = None):
        layer = lasagne.layers.InputLayer(shape=self._shape, input_var=input_var)

        # TODO: swap this part out with conv layers
        # ======================================================================================
        size = self.input_size
        while size > self.enc_size * 2:
            layer = lasagne.layers.DenseLayer(layer, num_units=size, nonlinearity=leaky_rectify)
            layer = lasagne.layers.DenseLayer(layer, num_units=size, nonlinearity=leaky_rectify)
            size = size // 2
        # ======================================================================================

        layer = lasagne.layers.DenseLayer(layer, num_units=self.enc_size, nonlinearity=tanh)
        return layer

    def _build_decoder(self, input_layer):
        layer = input_layer

        # TODO: swap this part out with deconv layers
        # ======================================================================================
        size = self.enc_size * 2
        while size < self.input_size * 2:
            layer = lasagne.layers.DenseLayer(layer, num_units=size, nonlinearity=leaky_rectify)
            layer = lasagne.layers.DenseLayer(layer, num_units=size, nonlinearity=leaky_rectify)
            size *= 2
        # ======================================================================================

        layer = lasagne.layers.DenseLayer(layer, num_units=self.input_size, nonlinearity=rectify)
        layer = lasagne.layers.ReshapeLayer(layer, shape=(-1, self.input_size))

        return layer

    def _build_discriminator(self, input_layer):
        layer = input_layer

        # TODO: vary the power of the discriminator
        # ======================================================================================
        k = 2 # some hyperparam
        for i in xrange(k):
            layer = DenseLayer(layer, num_units=self.output_size*2, nonlinearity=leaky_rectify)
        # ======================================================================================

        layer = DenseLayer(layer, num_units=self.output_size, nonlinearity=softmax)
        layer = lasagne.layers.ReshapeLayer(layer, shape=(-1, self.output_size))

        return layer


if __name__ == '__main__':

    if len(sys.argv) >= 5:
        # hyperparams
        ENCODING_SIZE = int(sys.argv[3])
        RECON_ALPHA = float(sys.argv[4])

        # other constants
        NUM_EPOCHS = int(sys.argv[1])
        BATCH_SIZE = int(sys.argv[2])
    else:
        print("Expects args: ")
        print("redaction.py <int>NUM_EPOCHS <int>BATCH_SIZE <int>ENCODING_SIZE <float>RECON_ALPHA")
        exit()

    # # hyperparams
    # ENCODING_SIZE = 100
    # RECON_ALPHA = 0.5
    # # other constants
    # NUM_EPOCHS = 1
    # BATCH_SIZE = 4

    X_train = np.loadtxt('data/X_train.txt')
    y_train = np.loadtxt('data/y_train.txt')
    X_test = np.loadtxt('data/X_test.txt')
    y_test = np.loadtxt('data/y_test.txt')

    inputs = T.matrix('inputs')
    targets = T.matrix('targets')
    r = RedactionNetwork(inputs, targets, shape=(None, X_train.shape[1]), encoding_size=ENCODING_SIZE,
                         output_size=y_train.shape[1], alpha=RECON_ALPHA)

    print("Starting Training ...")
    for epoch in xrange(NUM_EPOCHS):
        train_err = 0.0
        adv_acc = 0.0
        train_batches = 0.0

        t = time.time()
        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
            input, target = batch

            train_err += r.joint_train_fn(input, target)
            adv_acc += r.adv_accuracy_fn(input, target)
            train_batches += 1
        et = time.time()

        print("Epoch {} of {}: {:.4f} sec".format(epoch, NUM_EPOCHS, et - t))
        print("  training loss: \t{:.6f}".format((train_err)/train_batches))
        print("  prediction accuracy: \t{:.4f}%".format(((adv_acc)/train_batches)*100))

    name_str = str(sys.argv[0].split('.')[0]) + '_' + str(ENCODING_SIZE) + '_' + str(RECON_ALPHA)
    np.savez(name_str + '_params.npz', *r.joint_params)

    '''
    with np.load('model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    '''
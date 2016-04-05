"""
Character-level Convolutional Neural Networks

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
- https://github.com/yoonkim/CNN_sentence
"""
import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.typed_list

class HiddenLayer(object):
    """
    Class for HiddenLayer
    """
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None, use_bias=False):
        self.input = input
        self.activation = activation
        if W is None:            
            if activation.func_name == "ReLU":
                W_values = numpy.asarray( \
                    0.01 * rng.standard_normal(size=(n_in, n_out)), \
                     dtype=theano.config.floatX)
            else:                
                W_values = numpy.asarray(
                	rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), \
                    high=numpy.sqrt(6. / (n_in + n_out)), \
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')        
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')
        self.W = W
        self.b = b
        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)
        self.output = (lin_output if activation is None else activation(lin_output))
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


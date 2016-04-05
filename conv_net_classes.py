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

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

class LogisticRegression(object):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # parameters of the model
        self.params = [self.W, self.b]
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
		.. math::
		\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
		\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
		\ell (\theta=\{W,b\}, \mathcal{D})
		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
		correct label
		Note: we use the mean instead of the sum so that
		the learning rate is less dependent on the batch size
		"""
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
	    zero one loss over the size of the minibatch
	    :type y: theano.tensor.TensorType
	    :param y: corresponds to a vector that gives for each example the
	    correct label
	    """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class MLPDropout(object):
    """A multilayer perceptron with dropout"""
    def __init__(self,rng,input,layer_sizes,dropout_rates,activations,use_bias=True):
        #rectified_linear_activation = lambda x: T.maximum(0.0, x)
        # Set up all the hidden layers
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        next_layer_input = input
        #first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output
            # Reuse the parameters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)
        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)
        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors
        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]
    def predict(self, new_data):
        next_layer_input = new_data
        for i,layer in enumerate(self.layers):
            if i<len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred
    def predict_p(self, new_data):
        next_layer_input = new_data
        for i,layer in enumerate(self.layers):
            if i<len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        return p_y_given_x

class CharConvPoolLayer(object):
    """ Layer of a convolutional network """
    def __init__(self, rng, input, filter_shape, non_linear="tanh"):
        """
        filter_shape = [ cl_u_0, cl_u_1, d_chr, k_chr, d_wrd, k_wrd, hl_u, d_wrd, V_wrd, d_chr, V_chr ]
        map_shapes = [ word_idx_map_len, char_idx_map ] 
        Allocate a CharConvPoolLayer with shared variable internal parameters.
        :type input: lists of lists of lists ??
        :param input: a blog with multiple sentences
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 8
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)
        """
        self.input = input
        self.non_linear = non_linear

        # receives filter shapes
        self.cl_u_0 = filter_shape[0] # set by us = 1
        self.cl_u_1 = filter_shape[1] # set by us = 1

        self.k_chr = filter_shape[2] # 2
        self.k_wrd = filter_shape[3] # 3

        self.d_wrd = filter_shape[4] # 2 
        self.d_chr = filter_shape[5] # 1

        self.V_wrd = filter_shape[6] # len_word_dict
        self.V_chr = filter_shape[7] # len_char_dict

        """
        # initialize weights with random values
        self.W0 = theano.shared(numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.cl_u_0,self.d_chr*self.k_chr)), dtype=theano.config.floatX),name="W0")
        self.W1 = theano.shared(numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.cl_u_1,(self.d_wrd+self.cl_u_0)*self.k_wrd)), 
                                                dtype=theano.config.floatX),name="W1")
        self.W_wrd = theano.shared(numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.d_wrd,self.V_wrd)), 
                                                dtype=theano.config.floatX),name="W_wrd")
        self.W_chr = theano.shared(numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.d_chr,self.V_chr)), 
                                                dtype=theano.config.floatX),name="W_chr")
 
        # initialize biases with random values
        b0_values = numpy.zeros((self.cl_u_0,), dtype=theano.config.floatX)
        self.b0 = theano.shared(value=b0_values, name="b0")

        b1_values = numpy.zeros((self.cl_u_1,), dtype=theano.config.floatX)
        self.b1 = theano.shared(value=b1_values, name="b1")
        """
        # initialize weights with random values
        self.W0 = numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.cl_u_0,self.d_chr*self.k_chr)), dtype=theano.config.floatX)
        self.W1 = numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.cl_u_1,(self.d_wrd+self.cl_u_0)*self.k_wrd)), 
                                                dtype=theano.config.floatX)
        self.W_wrd = numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.d_wrd,self.V_wrd)), 
                                                dtype=theano.config.floatX)
        self.W_chr = numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(self.d_chr,self.V_chr)), 
                                                dtype=theano.config.floatX)
        # initialize biases with random values
        self.b0  = numpy.zeros((self.cl_u_0,), dtype=theano.config.floatX)

        self.b1 = numpy.zeros((self.cl_u_1,), dtype=theano.config.floatX)

        print('shape of W0 ' + str(self.W0.shape) )
        print('shape of W1 ' + str(self.W1.shape) )
        print('shape of W_wrd ' + str(self.W_wrd.shape) )
        print('shape of W_chr ' + str(self.W_chr.shape) )
        print('shape of b0 ' + str(self.b0.shape) )
        print('shape of b1 ' + str(self.b1.shape) )

        # convolve input feature maps with filters
        self.max_r_blog_list = []
        for sent in input:
            list_of_r_blog = []
            # word windows
            for word_windows in sent:
                # will be concatenated repeatedly
                list_of_r_blog.append( self.compute_r_sent(word_windows) )
            max_r_blog = list_of_r_blog[0]
            self.max_r_blog_list.append( max_r_blog )

    def get_one_hot_vec(self,v_values,dim):
        one_hot = np.zeros(dim) 
        one_hot[v_values] = 1
        one_hot_vec = np.asarray(one_hot,dtype=theano.config.floatX)
        return one_hot_vec

    def compute_max_r_wrd(self,char_windows):
        list_of_r_wch = []
        # one char window
        for char_window in char_windows:

            # initialize z_m
            z_m = None
            for char_vec in char_window:
                v_c = self.get_one_hot_vec( char_vec, self.V_chr)
                r_chr = self.W_chr.dot(v_c)
                #print( 'r_chr is ' + str(r_chr.shape) )
                if z_m:
                    z_m = np.concatenate((z_m,r_chr),axis=1)
                else:
                    z_m = r_chr
            #print('z_m ' + str(z_m.shape) )
            r_wch = z_m.reshape( (z_m.shape[0],1) ).dot(self.W0) + self.b0 
            list_of_r_wch.append( r_wch )
        max_r_wrd = list_of_r_wch[0] # return the first for now
        return max_r_wrd

    def compute_r_sent(self,word_windows):
        z_n = None
        # word, char windows for each word
        for (v_w_values,char_windows) in word_windows:
            # initialize v_w
            v_w = self.get_one_hot_vec(v_w_values, self.V_wrd)
            # weight * one-hot
            r_wrd = self.W_wrd.dot(v_w) 
            max_r_wrd = self.compute_max_r_wrd(char_windows)
            #print('max_r_wrd '+str(max_r_wrd.shape) )
            #print('r_wrd '+str(r_wrd.shape) )
            # concatenate embeddings (word + char)  
            u_n = np.concatenate((r_wrd.reshape( (r_wrd.shape[0],1) ),max_r_wrd),axis=0) 
            # joining embeddigns
            if not(z_n==None):
                z_n = np.concatenate((z_n,u_n),axis=0)
            else:
                z_n = u_n
        r_sent = z_n.reshape( (z_n.shape[0],1) ).dot(self.W1) + self.b1     
        return r_sent

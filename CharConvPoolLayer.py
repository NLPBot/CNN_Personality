import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class CharConvPoolLayer(object):
    """ Layer of a convolutional and pooling network """
    def __init__(self, rng, input, filter_shape, non_linear="tanh"):
        """
        filter_shape = [ cl_u_0, cl_u_1, d_chr, k_chr, d_wrd, k_wrd, hl_u, d_wrd, V_wrd, d_chr, V_chr ]
        Allocate a CharConvPoolLayer with shared variable internal parameters.
        :type input: lists of lists of lists ??
        :param input: a blog with multiple sentences
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
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

        # initialize weights with random values
        self.W0 = theano.shared(rng.randn(self.cl_u_0,self.d_chr*self.k_chr), 
        										dtype=theano.config.floatX, name='W0')
        self.W1 = theano.shared((rng.randn(self.cl_u_1,(self.d_wrd+self.cl_u_0)*self.k_wrd), 
                                                dtype=theano.config.floatX, name='W1')
        self.W_wrd = theano.shared((rng.randn(self.d_wrd,self.V_wrd), 
                                                dtype=theano.config.floatX, name='W_wrd')
        self.W_chr = theano.shared((rng.randn(self.d_chr,self.V_chr), 
                                                dtype=theano.config.floatX, name='W_chr')
        # initialize biases with random values
        self.b0  = theano.shared((self.cl_u_0,), dtype=theano.config.floatX, name='b0')

        self.b1 = theano.shared((self.cl_u_1,), dtype=theano.config.floatX, name='b1')

"""
        print('shape of W0 ' + str(self.W0.shape) )
        print('shape of W1 ' + str(self.W1.shape) )
        print('shape of W_wrd ' + str(self.W_wrd.shape) )
        print('shape of W_chr ' + str(self.W_chr.shape) )
        print('shape of b0 ' + str(self.b0.shape) )
        print('shape of b1 ' + str(self.b1.shape) )
"""
        # convolve input feature maps with multiple filters
        self.max_r_blog_list = []
        for sent in input:
            r_sent_list = [ self.compute_r_sent(word_windows) for word_windows in sent ]
            max_r_blog = r_sent_list[0] # find max
            self.max_r_blog_list.append( max_r_blog )

    def get_one_hot_vec(self,v_values,dim):
        one_hot = np.zeros(dim) 
        one_hot[v_values] = 1
        one_hot_vec = np.asarray(one_hot,dtype=theano.config.floatX)
        return one_hot_vec

    def compute_r_wch(self,char_window):

        # initialize z_m
        z_m = T.fvector('z_m')

        # v_c = theano.shared(char_window[i])
        # r_chr = T.dot(self.W_chr,v_c)
        z_m, updates = theano.scan(lambda i,z_m,W_chr,char_window: T.concatenate([z_m,T.dot(self.W_chr, \
        				theano.shared(self.get_one_hot_vec(char_window[i],self.V_chr)) )],axis=0),
        							sequences=i,
        							non_sequences=[z_m,W_chr,char_window])
        # append to list
        r_wch = T.dot(self.W0,z_m) + self.b0 
        return r_wch # return first for now

    def get_u_n(self,word_window):

    	# get params
    	(v_w_values,char_windows) = word_window 

	    # initialize v_w
	    v_w = theano.shared(self.get_one_hot_vec(v_w_values,self.V_wrd))

	    # weight * one-hot
	    r_wrd = T.dot(self.W_wrd,v_w)
	    r_wch_list = [ self.compute_r_wch(char_window) for char_window in char_windows ]
	    r_wch = r_wch_list[0] # find max

	    # concatenate embeddings (word + char) 
	    u_n = T.concatenate([r_wrd,r_wch],axis=0) 
	    return u_n

    def compute_r_sent(self,word_windows):

        z_n = fvector('z_n')
        z_n, updates = theano.scan(lambda i,z_n,word_windows: T.concatenate([z_n,get_u_n(word_windows[i])],axis=0),
        							sequences=i,
        							non_sequences=[z_n,word_windows])

        r_sent = T.dot(self.W1,z_n) + self.b1
        return r_sent
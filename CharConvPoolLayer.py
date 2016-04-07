import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

theano.config.compute_test_value = 'warn'

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
        self.W0 = theano.shared(rng.randn(self.cl_u_0,self.d_chr*self.k_chr), name='W0')
        self.W1 = theano.shared(rng.randn(self.cl_u_1,(self.d_wrd+self.cl_u_0)*self.k_wrd), name='W1')
        self.W_wrd = theano.shared(rng.randn(self.V_wrd,self.d_wrd), name='W_wrd')
        self.W_chr = theano.shared(rng.randn(self.V_chr,self.d_chr), name='W_chr')
        # initialize biases with random values
        self.b0  = theano.shared(rng.randn(self.cl_u_0,), name='b0')
        self.b1 = theano.shared(rng.randn(self.cl_u_1,), name='b1')

        print('shape of W0 ' + str(self.W0.shape.eval() ) )
        print('shape of W1 ' + str(self.W1.shape.eval()) )
        print('shape of W_wrd ' + str(self.W_wrd.shape.eval()) )
        print('shape of W_chr ' + str(self.W_chr.shape.eval()) )
        print('shape of b0 ' + str(self.b0.shape.eval()) )
        print('shape of b1 ' + str(self.b1.shape.eval()) )

        # convolve input feature maps with multiple filters
        self.max_r_sent_list = []
        for sent in input:
            r_sent_list = [ self.compute_r_sent(word_windows) for word_windows in sent ]
            max_r_sent = r_sent_list[0] # find max
            self.max_r_sent_list.append( max_r_sent )

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
        #T.dot(self.W_chr, theano.shared(self.get_one_hot_vec(char,self.V_chr)) )
        z_m_list = [ self.W_chr[char] for char in char_window ]
        z_m = T.concatenate( [z_m_list], axis=0 )

        # append to list
        """
        self.print_dim('self.W0',self.W0)
        self.print_dim('z_m',z_m)
        self.print_dim('T.dot(z_m,self.W0)',T.dot(z_m,self.W0))
        """
        r_wch = T.dot(self.W0,z_m) + self.b0 
        return r_wch # return first for now

    def get_u_n(self,word_window):

    	# get params
    	(v_w_values,char_windows) = word_window 

        # initialize v_w
        #v_w = theano.shared(self.get_one_hot_vec(v_w_values,self.V_wrd))

        # weight * one-hot
        #r_wrd = T.dot(self.W_wrd,v_w)
        r_wrd = self.W_wrd[v_w_values]
        r_wch_list = [ self.compute_r_wch(char_window) for char_window in char_windows ]
        r_wch = r_wch_list[0] # find max

        # concatenate embeddings (word + char) 
        """
        self.print_dim('r_wrd',r_wrd)
        self.print_dim('r_wch',r_wch)
        """
        #self.print_dim('r_wch[0]',r_wch[0])
        u_n = T.concatenate([r_wrd.T,r_wch],axis=0) 

        #self.print_dim('u_n',u_n)

        return [r_wrd.T,r_wch]

    def compute_r_sent(self,word_windows):

        z_n = T.fvector('z_n')
        z_n_list = []
        word_windows_list = [ z_n_list.extend(self.get_u_n(word_window)) for word_window in word_windows ]
        z_n = T.concatenate( z_n_list, axis=0 )
        """
        self.print_dim('self.W1',self.W1)
        self.print_dim('z_n',z_n)
        self.print_val('z_n',z_n)
        """
        r_sent = T.dot(self.W1,z_n) + self.b1 
        self.print_val('r_sent',r_sent)
        return r_sent

    def print_dim(self,name,val):
    	print( name + ' ' + str(val.shape.eval()) )

    def print_val(self,name,val):
    	print( name + ' ' + str(val.eval()) )
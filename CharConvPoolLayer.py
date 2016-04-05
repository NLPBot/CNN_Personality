import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

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
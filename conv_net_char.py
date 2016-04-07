"""
Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
import string
from CharConvPoolLayer import *

warnings.filterwarnings("ignore")  

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def get_filter_shape(len_word_dict,len_char_dict):
    """
        self.cl_u_0 = filter_shape[0] # set by us = 1
        self.cl_u_1 = filter_shape[1] # set by us = 1

        self.k_chr = filter_shape[2] # 1
        self.k_wrd = filter_shape[3] # 2

        self.d_wrd = filter_shape[4] # 2 
        self.d_chr = filter_shape[5] # 1

        self.V_wrd = filter_shape[6] # len_word_dict
        self.V_chr = filter_shape[7] # len_char_dict

    """
    filter_shape = [1]*8
    filter_shape[0] = 1 # set by us = 1
    filter_shape[1] = 1 # set by us = 1
    filter_shape[2] = 2 # set by us = 2
    filter_shape[3] = 3 # 3
    filter_shape[4] = 2 # set by us = 2
    filter_shape[5] = 1 # set by us = 1
    filter_shape[6] = len_word_dict # len_word_dict
    filter_shape[7] = len_char_dict # len_char_dict
    return filter_shape

def train_conv_net(datasets,U,conv_non_linear,len_word_dict,len_char_dict):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)  
    #define model architecture
    # parameters: rng, input, filter_shape, non_linear="tanh"
    filter_shape = get_filter_shape(len_word_dict,len_char_dict)
    conv_layer = CharConvPoolLayer(rng,datasets[0],filter_shape,non_linear=conv_non_linear)

    #for r_sent in conv_layer.max_r_sent_list:


    # get cost
    """
    # probability that target = 1
    prediction = p_1 > 0.5 # the prediction threshold
    """

    # cost function
    """
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
    cost = xent.mean() + 0.01 * (w_1**2).sum()
    gw_1, gb_1, gw_2, gb_2 = T.grad(cost,[w_1,b_1,w_2,b_2])
    """


    # prediction function
    """
    predict = theano.function(inputs = [x], outputs = prediction)
    """


    # training function
    """
    train = theano.function(
                        inputs = [x,y], 
                        outputs = [prediction, xent],
                        updates = {w_1 : w_1-0.1*gw_1, b_1 : b_1-0.1*gb_1,
                                    w_2 : w_2-0.1*gw_2, b_2 : b_2-0.1*gb_2})

    """



    # training

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                                        borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                                        borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def get_char_idx_map():
    """
    Initialize char_idx_map, 100 chars in total
    """
    char_idx_map = dict()
    all_letters = dict.fromkeys(string.ascii_letters,0)
    all_digits = dict.fromkeys(string.digits,0)
    all_puncs = dict.fromkeys(string.punctuation,0)
    #all_whitespace = dict.fromkeys(string.whitespace,0)
    char_idx_map.update(all_letters)
    char_idx_map.update(all_digits)
    char_idx_map.update(all_puncs)
    #char_idx_map.update(all_whitespace)
    i = 1
    for char in char_idx_map.keys():
        char_idx_map[char] = i
        i += 1
    return char_idx_map

def get_char_idx_from_sent(sent, char_idx_map, window_size=2):
    """
    Transforms sentence into a windows of list indices. Pad with zeroes.
    """
    x = []
    for chars_window in range(len(list(sent)[::window_size])):
        char_indices_in_window = []
        for char in list(sent)[chars_window:chars_window+window_size]:
            if char in char_idx_map:
                char_indices_in_window.append(char_idx_map[char])
        x.append( char_indices_in_window )
    return x

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. 
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, word_window_len=3, char_window_len=2):
    """
    Transforms sentences into matrices.
    """
    train, test = [], []
    ## Init char_idx_map
    char_idx_map = get_char_idx_map()
    len_char_dict = len( char_idx_map.keys() )
    for rev in revs[:1]:
        label = rev["y"]
        sent = rev["text"]
        sent_embeddings = []
        for start_word_index in range(len(sent.split()))[::word_window_len]:
            word_window = sent.split()[start_word_index:start_word_index+word_window_len]
            word_window_embeddings = []
            for word in word_window:
                v_w = get_idx_from_sent(word, word_idx_map)
                v_c_tuple_list = get_char_idx_from_sent(word, char_idx_map, window_size=char_window_len)
                word_char_embeddings = ( v_w, v_c_tuple_list )
                word_window_embeddings.append( word_char_embeddings )
            sent_embeddings.append(word_window_embeddings)
        #print(sent_embeddings)
        if rev["split"]==cv:            
            test.append(sent_embeddings)        
        else:  
            train.append(sent_embeddings)   
    return [train, test], len_char_dict

if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"  
    print "model architecture: CNN-static"
    non_static=False
    execfile("CharConvPoolLayer.py")    
    print "using: word2vec vectors" 
    U = W
    results = []
    r = range(0,10)    
    for i in r: 
        datasets, len_char_dict = make_idx_data_cv(revs, word_idx_map, i, word_window_len=3, char_window_len=2)
        perf = train_conv_net(datasets,U,"relu",len(word_idx_map.keys()),len_char_dict)
        #print "cv: " + str(i) + ", perf: " + str(perf)
        #results.append(perf)  
        
    #print str(np.mean(results))

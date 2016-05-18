import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class MultiLayerConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:
  
  INPUT -> [CONV -> RELU -> POOL]*2 -> [FC -> RELU] -> FC -> SOFTMAX

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32),
               num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    (C, H, W) = input_dim
    K = num_filters
    (F_conv, P_conv, S_conv) = (filter_size, (filter_size - 1) / 2, 1)
    (F_pool, P_pool, S_pool) = (2, 0, 2)
    
    self.params = {}

    # Layer counter
    i_layer = 1
    
    #
    # NOTE: In the following, due to parameter sharing,
    #       we don't actually need the W_out, H_out, D_out
    #       size parameters to generate the weights in the initial
    #       CONV layers.  They are, however, needed to calculate
    #       the number of weights necessary for the FC layer
    #       after the last POOL.
    #

    #
    # CONV -> RELU -> POOL
    #
    (W_in, H_in, D_in) = (W, H, C)
    print "INPUT: [%dx%dx%d]  weights: 0" % (W_in, H_in, D_in)
    self.params['W' + str(i_layer)] = weight_scale * np.random.randn(K, C, F_conv, F_conv)
    self.params['b' + str(i_layer)] = np.zeros(num_filters)
    W_conv_out = 1 + (W_in - F_conv + (2 * P_conv)) / S_conv
    H_conv_out = 1 + (H_in - F_conv + (2 * P_conv)) / S_conv
    D_conv_out = K
    print "CONV: [%dx%dx%d]  weights: (%d*%d*%d)*%d" % (W_conv_out, H_conv_out, D_conv_out, F_conv, F_conv, C, K)
    W_pool_out = 1 + (W_conv_out - F_pool) / S_pool
    H_pool_out = 1 + (H_conv_out - F_pool) / S_pool
    D_pool_out = D_conv_out
    print "POOL: [%dx%dx%d]  weights: 0" % (W_pool_out, H_pool_out, D_pool_out)
    (W_out, H_out, D_out) = (W_pool_out, H_pool_out, D_pool_out)
    i_layer += 1

    #
    # CONV -> RELU -> POOL
    #
    (W_in, H_in, D_in) = (W_out, H_out, D_out)
    self.params['W' + str(i_layer)] = weight_scale * np.random.randn(K, D_in, F_conv, F_conv)
    self.params['b' + str(i_layer)] = np.zeros(num_filters)
    W_conv_out = 1 + (W_in - F_conv + (2 * P_conv)) / S_conv
    H_conv_out = 1 + (H_in - F_conv + (2 * P_conv)) / S_conv
    D_conv_out = K
    print "CONV: [%dx%dx%d]  weights: (%d*%d*%d)*%d" % (W_conv_out, H_conv_out, D_conv_out, F_conv, F_conv, D_in, K)
    W_pool_out = 1 + (W_conv_out - F_pool) / S_pool
    H_pool_out = 1 + (H_conv_out - F_pool) / S_pool
    D_pool_out = D_conv_out
    print "POOL: [%dx%dx%d]  weights: 0" % (W_pool_out, H_pool_out, D_pool_out)
    (W_out, H_out, D_out) = (W_pool_out, H_pool_out, D_pool_out)
    i_layer += 1

    #
    # FC -> RELU
    #
    (W_in, H_in, D_in) = (W_out, H_out, D_out)
    self.params['W' + str(i_layer)] = weight_scale * np.random.randn(W_in * H_in * D_in, hidden_dim)
    self.params['b' + str(i_layer)] = np.zeros(hidden_dim)
    print "FC: [1x1x%d]  weights: %d*%d*%d*%d" % (hidden_dim, W_in, H_in, D_in, hidden_dim)
    i_layer += 1

    #
    # FC -> SOFTMAX
    #
    self.params['W' + str(i_layer)] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b' + str(i_layer)] = np.zeros(num_classes)
    print "FC: [1x1x%d]  weights: %d*%d" % (num_classes, hidden_dim, num_classes)
    i_layer += 1

    # Find the number of layers
    num_layers = i_layer - 1

    # Print the layer shapes for debugging purposes
    # for i_layer in np.arange(1,num_layers+1):
    #   print "W" + str(i_layer) + ".shape = ", self.params['W' + str(i_layer)].shape
    #   print "b" + str(i_layer) + ".shape = ", self.params['b' + str(i_layer)].shape

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the multi-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    c1, conv_relu_pool_1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    c2, conv_relu_pool_2_cache = conv_relu_pool_forward(c1, W2, b2, conv_param, pool_param)
    h1, affine_relu_cache = affine_relu_forward(c2, W3, b3)
    scores, softmax_cache = affine_forward(h1, W4, b4)

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores,y)
    
    # Add regularization to loss.
    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)
    loss += 0.5 * self.reg * np.sum(W3 * W3)
    loss += 0.5 * self.reg * np.sum(W4 * W4)
    
    # Do the backwards pass
    dh1, dW4, db4 = affine_backward(dscores, softmax_cache)
    dc2, dW3, db3 = affine_relu_backward(dh1, affine_relu_cache)
    dc1, dW2, db2 = conv_relu_pool_backward(dc2, conv_relu_pool_2_cache)
    dx, dW1, db1 = conv_relu_pool_backward(dc1, conv_relu_pool_1_cache)
    
    # Add regularization to the weight gradients
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4

    # Store
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    grads['W4'] = dW4
    grads['b4'] = db4
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

#
# NOTE: This class is under construction/possibly abandonded!
#
class DeepConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:
  
  INPUT -> [[CONV -> RELU] * N_ -> POOL?] * M_ -> [FC -> RELU] * K_ -> FC -> SOFTMAX

  where the variables are shorthand for the following parameters:
  N = num_inner_conv_layers
  M = num_outer_conv_layers
  K = num_affine_layers
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32),
               num_inner_conv_layers=1, num_outer_conv_layers=1, use_pooling=True, num_affine_layers=1,
               num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_conv_layers: M1, i.e. the number of [conv-relu-conv-relu-pool] layers.
    - num_affine_layers: M2, i.e. the number of [affine] layers.
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    (C, H, W) = input_dim
    K = num_filters
    F = filter_size
    F_conv = filter_size
    S_conv = 1
    P_conv = 1
    F_pool = 2
    S_pool = 2
    P_pool = 0
    N_ = num_inner_conv_layers
    M_ = num_outer_conv_layers
    K_ = num_affine_layers
    
    self.params = {}

    # Total layer counter
    i_layer = 1

    # Initialise input volume dimensions
    W_in = W
    H_in = H
    D_in = C

    # Loop through [CONV -> RELU] * N_
    for i_outer_layer in np.arange(1,M_+1):

      # Loop through [[CONV -> RELU] * N_ -> POOL?] * M_ 
      for i_inner_layer in np.arange(1,N_+1):
       
        # Output volume dimensions for CONV -> RELU
        #
        # NOTE: Thanks to parameter sharing, we won't actually
        # need these for the CONV layer, but we will for the pool layer.
        #
        # W_out = 1 + (W_in - F_conv + (2 * P_conv)) / S_conv
        # H_out = 1 + (H_in - F_conv + (2 * P_conv)) / S_conv
        # D_out = K
        
        # Build weights for CONV -> RELU
        self.params['W' + str(i_layer)] = weight_scale * np.random.randn(K, C, F, F)
        self.params['b' + str(i_layer)] = np.zeros(K)

        # Update input volume dimensions
        # W_in = W_out
        # H_in = H_out
        # D_in = D_out

        # Increment layer counter
        i_layer += 1

      if use_pooling:
        
        # Output volume dimensions for POOL
        # W_out = 1 + (W_in - F_pool) / S_pool
        # H_out = 1 + (H_in - F_pool) / S_pool
        # D_out = D_in

        # Build weights for POOL
        self.params['W' + str(i_layer)] = weight_scale * np.random.randn(K, C, F, F)
        self.params['b' + str(i_layer)] = np.zeros(K)

        # Update input volume dimensions
        # W_in = W_out
        # H_in = H_out
        # D_in = D_out
        
        # Increment layer counter
        i_layer += 1



    # Find the number of layers
    num_layers = i_layer - 1
    
    # Print the layer shapes for debugging purposes
    for i_layer in np.arange(1,num_layers+1):
      print "W" + str(i_layer) + ".shape = ", self.params['W' + str(i_layer)].shape
      print "b" + str(i_layer) + ".shape = ", self.params['b' + str(i_layer)].shape

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the multi-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    c1, conv_relu_pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    h1, affine_relu_cache = affine_relu_forward(c1, W2, b2)
    scores, softmax_cache = affine_forward(h1, W3, b3)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores,y)
    
    # Add regularization to loss.
    loss += 0.5 * self.reg * np.sum(W1 * W1)
    loss += 0.5 * self.reg * np.sum(W2 * W2)
    loss += 0.5 * self.reg * np.sum(W3 * W3)
    
    # Do the backwards pass
    dh1, dW3, db3 = affine_backward(dscores, softmax_cache)
    dc1, dW2, db2 = affine_relu_backward(dh1, affine_relu_cache)
    dx, dW1, db1 = conv_relu_pool_backward(dc1, conv_relu_pool_cache)
    
    # Add regularization to the weight gradients
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3

    # Store
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

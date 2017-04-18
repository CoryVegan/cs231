import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A convolutional network which can form any architecture
  consisting of any number of the following hidden layers:

  conv - relu, conv - relu - 2x2 max pool, and affine - relu

  And an output layer of:

  affine - softmax

  Note: we assume that affine layers never precede conv layers

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), affine_dim=100,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               hidden_layers=[{'name': 'conv'}, {'name': 'pool'},
                              {'name': 'affine'}],
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in every convolutional layer
    - filter_size: Size of filters to use in every convolutional layer
    - affine_dim: Number of units to use in every fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - hidden_layers: Strings defining the network architecture
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #self.num_filters = num_filters
    #self.filter_size = filter_size
    self.affine_dim = affine_dim
    self.num_classes = num_classes

    layers = hidden_layers[:]
    layers.append({'name': 'output'})
    self.layers = layers

    for i, layer in enumerate(self.layers):
      if layer['name'] == 'conv':
        D, H, W  = input_dim

        filter_size = layer.get('filter_size', 3)
        num_filters = layer.get('num_filters', 32)
        stride = layer.get('stride', 1)
        pad = layer.get('pad', (filter_size - 1) / 2)
        #stride = 1

        #FH, FW = filter_size, filter_size
        #F = num_filters

        shape = (num_filters, D, filter_size, filter_size)
        self.params['W' + str(i + 1)] = weight_scale * np.random.randn(*shape)
        self.params['b' + str(i + 1)] = np.zeros(num_filters)

        self.params['gamma' + str(i + 1)] = np.ones(num_filters)
        self.params['beta' + str(i + 1)] = np.zeros(num_filters)

        output_H = (H + 2 * pad - filter_size) / stride + 1
        output_W = (W + 2 * pad - filter_size) / stride + 1
        output_D = num_filters
        #output_D = F

        input_dim = (output_D, output_H, output_W)

      elif layer['name'] == 'pool':
        pool_size = layer.get('pool_size', 2)
        pool_stride = layer.get('stride', 2)

        output_H = (output_H - pool_size) / pool_stride + 1
        output_W = (output_W - pool_size) / pool_stride + 1

        input_dim = (output_D, output_H, output_W)

      elif layer['name'] == 'affine':
        num_neurons = np.prod(input_dim)
        shape = (num_neurons, affine_dim)

        self.params['W' + str(i + 1)] = weight_scale * np.random.randn(*shape)
        self.params['b' + str(i + 1)] = np.zeros(affine_dim)

        self.params['gamma' + str(i + 1)] = np.ones(affine_dim)
        self.params['beta' + str(i + 1)] = np.zeros(affine_dim)

        input_dim = affine_dim

      elif layer['name'] == 'output':
        shape = (affine_dim, num_classes)

        self.params['W' + str(i + 1)] = weight_scale * np.random.randn(*shape)
        self.params['b' + str(i + 1)] = np.zeros(num_classes)

      else:
        raise NameError('{} is not a valid layer name'.format(layer['name']))

    num_hidden_layers = len(hidden_layers)
    self.bn_params = [{'mode': 'train'} for i in xrange(num_hidden_layers)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    # pass conv_param to the forward pass for the convolutional layer
    #conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    #pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    caches = []
    scores = X.copy()

    for i, layer in enumerate(self.layers):
      if layer['name'] == 'pool':
        pool_size = layer.get('pool_size', 2)
        pool_stride = layer.get('stride', 2)

        pool_param = {'pool_height': pool_size, 'pool_width': pool_size,
                      'stride': pool_stride}
        scores, pool_c = max_pool_forward_fast(scores, pool_param)
        c = (conv_c, bn_c, relu_c, pool_c)

      # weight layer
      else:
        W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

        if layer['name'] == 'output':
          scores, affine_c = affine_forward(scores, W, b)
          c = affine_c

        # hidden layer
        else:
          gamma = self.params['gamma' + str(i + 1)]
          beta = self.params['beta' + str(i + 1)]
          bn_param = self.bn_params[i]

          if layer['name'] == 'conv':
            stride = layer.get('stride', 1)
            filter_size = layer.get('filter_size', 3)
            pad = layer.get('pad', (filter_size - 1) / 2)

            conv_param = {'stride': stride, 'pad': pad}
            scores, conv_c = conv_forward_fast(scores, W, b, conv_param)
            scores, bn_c = spatial_batchnorm_forward(scores, gamma, beta, bn_param)
            scores, relu_c = relu_forward(scores)
            c = (conv_c, bn_c, relu_c)

          # affine layer
          else:
            scores, affine_c = affine_forward(scores, W, b)
            scores, bn_c = batchnorm_forward(scores, gamma, beta, bn_param)
            scores, relu_c = relu_forward(scores)
            c = (affine_c, bn_c, relu_c)

      caches.append(c)

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
    loss, dscores = softmax_loss(scores, y)

    num_layers = len(self.layers)
    for i in xrange(num_layers - 1, -1 , -1):
      layer = self.layers[i]
      c = caches.pop()

      if layer['name'] == 'pool':
        conv_c, bn_c, relu_c, pool_c = c
        dscores = max_pool_backward_fast(dscores, pool_c)

      # weight layer
      else:
        if layer['name'] == 'output':
          affine_c = c
          dscores, dW, db = affine_backward(dscores, affine_c)

        # hidden layer
        else:
          if layer['name'] == 'conv':
            conv_c, bn_c, relu_c = c
            dscores = relu_backward(dscores, relu_c)
            dscores, dgamma, dbeta = spatial_batchnorm_backward(dscores, bn_c)
            dscores, dW, db = conv_backward_fast(dscores, conv_c)

          # affine layer
          else:
            affine_c, bn_c, relu_c = c
            dscores = relu_backward(dscores, relu_c)
            dscores, dgamma, dbeta = batchnorm_backward_alt(dscores, bn_c)
            dscores, dW, db = affine_backward(dscores, affine_c)

          grads['gamma' + str(i + 1)], grads['beta' + str(i + 1)] = dgamma, dbeta

        grads['W' + str(i + 1)], grads['b' + str(i + 1)] = dW, db

        W = self.params['W' + str(i + 1)]
        dW += self.reg * W
        loss += 0.5 * self.reg * np.sum(W**2)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads



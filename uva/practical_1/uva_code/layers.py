import numpy as np
"""
This module implements various layers for the network.
You should fill in code into indicated sections.
"""

class Layer(object):
  """
  Base class for all layers classes.

  """
  def __init__(self, layer_params = None):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer.

    """
    self.train_mode = False

  def initialize(self):
    """
    Cleans cache. Cache stores intermediate variables needed for backward computation.

    """
    self.cache = None

  def layer_loss(self):
    """
    Returns partial loss of layer parameters for regularization term of full loss.

    Returns:
      loss: Partial loss of layer parameters.

    """
    return 0.

  def set_train_mode(self):
    """
    Sets train mode for the layer.

    """
    self.train_mode = True

  def set_test_mode(self):
    """
    Sets test mode for the layer.

    """
    self.train_mode = False

  def forward(self, X):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    raise NotImplementedError("Forward pass is not implemented for base Layer class.")

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradient of the output with respect to the input of the layer.

    """
    raise NotImplementedError("Backward pass is not implemented for base Layer class.")

class LinearLayer(Layer):
  """
  Linear layer.

  """
  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer:
          input_size - input dimension;
          output_size - output dimension;
          weight_decay - regularization parameter for the weights;
          weight_scale - scale of normal distrubtion to initialize weights.

    """

    self.layer_params = layer_params
    self.layer_params.setdefault('weight_decay', 0.0)
    self.layer_params.setdefault('weight_scale', 0.001)

    self.params = {'w': None, 'b': None}
    self.grads = {'w': None, 'b': None}

    self.train_mode = False

  def initialize(self):
    """
    Initializes weights and biases. Cleans cache.
    Cache stores intermediate variables needed for backward computation.

    """
    ########################################################################################
    # Initialize weights self.params['w'] using normal distribution with mean = 0 and      #
    # std = self.layer_params['weight_scale'].                                             #
    #                                                                                      #
    # Initialize biases self.params['b'] with 0.                                           #
    ########################################################################################
    w_shape = (self.layer_params['input_size'], self.layer_params['output_size'])
    self.params['w'] = np.random.normal(0, self.layer_params['weight_scale'], w_shape)
    self.params['b'] = np.zeros((1, self.layer_params['output_size']))
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    self.cache = None

  def layer_loss(self):
    """
    Returns partial loss of layer parameters for regularization term of full loss.

    Returns:
      loss: Partial loss of layer parameters.

    """

    ########################################################################################
    # Compute the loss of the layer which responsible for L2 regularization term. Store it #
    # in loss variable.                                                                    #
    ########################################################################################
    loss = self.layer_params['weight_decay'] * self.params['w']
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    return loss

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # Implement forward pass for LinearLayer. Store output of the layer in out varible.    #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = x.dot(self.params['w']) + self.params['b']

    # Cache if in train mode
    if self.train_mode:
      self.cache = out
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    if not self.train_mode:
      raise ValueError("Backward is not possible in test mode")

    ########################################################################################
    # Implement backward pass for LinearLayer. Store gradient of the loss with respect to  #
    # layer parameters in self.grads['w'] and self.grads['b']. Store gradient of the loss  #
    # with respect to the input in dx variable.                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    dx = dout.dot(self.params['w'].T)
    self.grads['w'] = dx.T.dot(self.cache) + self.layer_loss()
    assert self.grads['w'].shape == self.params['w'].shape,
        '{} != {}'.format(self.grads['w'].shape, self.params['w'].shape)
    self.grads['b'] = dx.sum()
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class ReLULayer(Layer):
  """
  ReLU activation layer.

  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for ReLULayer. Store output of the layer in out variable.     #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for ReLULayer. Store gradient of the loss with respect to    #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class SigmoidLayer(Layer):
  """
  Sigmoid activation layer.

  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for SigmoidLayer. Store output of the layer in out variable.  #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for SigmoidLayer. Store gradient of the loss with respect to #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class TanhLayer(Layer):
  """
  Tanh activation layer.

  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for TanhLayer. Store output of the layer in out variable.     #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for TanhLayer. Store gradient of the loss with respect to    #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class ELULayer(Layer):
  """
  ELU activation layer.

  """

  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.

    Args:
      layer_params: Dictionary with parameters for the layer:
          alpha - alpha parameter;

    """
    self.layer_params = layer_params
    self.layer_params.setdefault('alpha', 1.0)
    self.train_mode = False

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for ELULayer. Store output of the layer in out variable.      #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for ELULayer. Store gradient of the loss with respect to     #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class SoftMaxLayer(Layer):
  """
  Softmax activation layer.

  """

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: Input to the layer.

    Returns:
      out: Output of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for SoftMaxLayer. Store output of the layer in out variable.  #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: Gradients of the previous layer.

    Returns:
      dx: Gradients with respect to the input of the layer.

    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for SoftMaxLayer. Store gradient of the loss with respect to #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################s
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

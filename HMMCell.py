import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

class TransitionMatrixChainInitializer(tf.keras.initializers.Initializer):
  def __init__(self, succScore=1.0):
    self.succScore = succScore

  def __call__(self, shape, dtype=None, **kwargs):
    if len(shape)<2 or shape[-2] != shape[-1]:
        raise ValueError('Last two dimensions of TransitionMatrixChainInitializer'
                         'must specify a square matrix.')
    if len(shape) != 3:
        # TODO remove this requirement later
        raise ValueError('TransitionMatrixChainInitializer requires 3 dims')
    n = shape[-1]
    u = shape[-3]

    # a tridiagonal band (per unit).
    diagonals = self.succScore * np.ones((u, 3, n), dtype=np.float32)
    A = tf.linalg.diag(diagonals, k = (-1, 1))
    return A

  def get_config(self):  # To support serialization
    return {"succScore": self.succScore}

class HMMCell(Layer):
  """Cell class for a HMM as a RNN.
  This class processes one step within the whole time sequence input.
  Arguments:
    n: positive integer number of hidden states, dimensionality of the output space.
  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`. feature=s is emission alphabet size
    states: A 2D tensor with shape of `[batch, n]`, which is the forward variable alpha from
      the previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
  Examples:
  ```python
  inputs = np.random.random([32, 3, 2]).astype(np.float32)
  hmmC = HMMCell(3)
  output = hmmC(inputs)  # The output has shape `[32, 3]`.
  hmm = tf.keras.layers.RNN(
      HMMCell(3),
      return_sequences = True,
      return_state = True)
  # whole_sequence_output has shape `[32, 3, 3]`.
  # final_state has shape `[32, 3]`.
  whole_sequence_output, final_state = hmm(inputs)
  ```
  """

  def __init__(self,
               units, # number of HMMs
               n, # number of HMM hidden states
               transition_initializer='random_uniform',
               emission_initializer='random_uniform',
               init_initializer='random_uniform',
               **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.n = n # number of HMM hidden states
    self.transition_initializer = initializers.get(transition_initializer)
    self.emission_initializer = initializers.get(emission_initializer)
    self.init_initializer = initializers.get(init_initializer)
    self.state_size = [1, self.n, 1]
    self.output_size = self.n

  def build(self, input_shape):
    self.s = input_shape[-1] # emission alphabet size
    self.emission_kernel = self.add_weight(
        shape=(self.units, self.n, self.s),
        initializer=self.emission_initializer,
        name='emission_kernel') # closely related to B
    self.transition_kernel = self.add_weight(
        shape=(self.units, self.n, self.n),
        initializer=self.transition_initializer,
        name='transition_kernel') # closely related to A
    self.init_kernel = self.add_weight(
        shape=(self.units, self.n),
        initializer=self.init_initializer,
        name='init_kernel') # closely related to initial distribution of first hidden state
    self.built = True

  def call(self, inputs, states, training=None):
    verbose = False
    old_is_init, old_forward, old_loglik = states
    batch_size = old_forward.shape[0]
    if verbose:
        print ("batch_size=", batch_size)
        print ("old_is_init=", old_is_init)
        print ("old_forward=\n", old_forward, " shape", old_forward.shape)
        print ("old_loglik=", old_loglik)

    I0 = tf.dtypes.cast(old_is_init, tf.float32)
    R0 = tf.tensordot(I0, self.I, axes=0)
    R1 = tf.linalg.matvec(self.A, old_forward, transpose_a=True)
    R = R0 + R1
    R = tf.identity(R, name="R")
    if verbose:
        print (f"R0:{R0}\nR1:{R1}\nR:{R}")
    #   [units, n, s]     [batch_size, s]
    # E has shape [batch_size, units, n]
    E = tf.linalg.matvec(tf.expand_dims(self.B, 0),
                         tf.expand_dims(inputs, 1), name="E")
    forward = tf.multiply(E, R, name="forward")
    S = tf.reduce_sum(forward, axis=-1, name="loglik")
    loglik = old_loglik + tf.math.log(S)
    forward = forward / tf.expand_dims(S, -1)
    batch_size = tf.shape(inputs)[0] # 'call' can be given None as batch size
    is_init = tf.zeros(batch_size, dtype='int8', name="is_init")
    new_state = [is_init, forward, loglik]
    new_state = [new_state] if nest.is_nested(states) else new_state
    if verbose:
        print ("new_state", new_state)
    return loglik, new_state

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """ initially the HMM starts in state 0 """
    is_init = tf.ones(batch_size, dtype='int8')
    forward = tf.zeros([batch_size, self.units, self.n], dtype=np.float32)
    loglik = tf.zeros([batch_size, self.units])
    S = [is_init, forward, loglik]
    return S
 
  def get_config(self):
    config = {
        'units': self.units,
        'n': self.n,
        'alphabet_size': self.s,
        'transition_initializer':
            initializers.serialize(self.transition_initializer),
        'emission_initializer':
            initializers.serialize(self.emission_initializer),
        'init_initializer':
            initializers.serialize(self.init_initializer)
    }
    # config.update(_config_for_enable_caching_device(self))
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
  # convert parameter matrices to stochastic matrices
  # TODO: this could be more efficient, maybe using tensorflow.python.keras.constraints?

  @property
  def A(self):
      transition_matrix = tf.nn.softmax(self.transition_kernel, axis=-1, name="A")
      return transition_matrix

  @property
  def B(self):
      emission_matrix = tf.nn.softmax(self.emission_kernel, axis=-1, name="B")
      return emission_matrix

  @property
  def I(self):
      initial_distribution = tf.nn.softmax(self.init_kernel, axis=-1, name="I")
      return initial_distribution

  def print_pars(self):
    with np.printoptions(precision=5, suppress=True, linewidth=100):
        print("transition matrices A:\n", self.A.numpy())
        # print("emission matrices B:\n", self.B.numpy())
        # print("initial distributions I:\n", self.I.numpy())





class HMMLayer(Layer):
  def __init__(self, num_hmms, num_hidden_states, posteriorFinalStateProbs=True):
    super(HMMLayer, self).__init__()
    self.units = num_hmms
    self.n = num_hidden_states
    self.posteriorFinalStateProbs = posteriorFinalStateProbs

  def build(self, input_shape):
    emi_ini = tf.keras.initializers.RandomUniform(minval=0, maxval=0.1)
    self.C = HMMCell(self.units, self.n,
                     emission_initializer=emi_ini,
                     transition_initializer=TransitionMatrixChainInitializer(3.))
    self.C.build(input_shape)
    self.F = tf.keras.layers.RNN(self.C, return_state = True)
    
  def call(self, input):
    alpha, _, lastcol, loglik = self.F(input)
    seqlen = input.shape[-2]
    if (seqlen is not None and seqlen > 0):
        # normalize to make comparable between lengths
        loglik = loglik / seqlen
    # standardize to make roughly comparable between alphabet sizes
    loglik += tf.math.log(tf.cast(self.C.s, tf.float32))
    if seqlen is not None:
    #    # this loss improves predictions when starting with a uniform, full transition matrix
        transhom = 1.0 - tf.reduce_sum(tf.math.square(self.C.A)) / (self.n * self.units)
        self.add_loss(0.1 * transhom)

    output = loglik
    if self.posteriorFinalStateProbs:
        shape = lastcol.shape
        batch_size = lastcol.shape[0]
        newshape = shape[0:-1]
        posteriorFinalStateProbs = tf.reshape(lastcol, [-1, self.units * self.n])
        output = tf.concat([loglik, posteriorFinalStateProbs], axis=-1)

    return output

  def get_config(self):
    config = {
        'units': self.units,
        'n': self.n
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

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
               n, # number of HMM hidden states, output size
               transition_initializer='random_uniform',
               emission_initializer='random_uniform',
               init_initializer='constant', # = uniformly on set of states
               **kwargs):
    super().__init__(**kwargs)
    self.n = n # number of HMM hidden states
    self.transition_initializer = initializers.get(transition_initializer)
    self.emission_initializer = initializers.get(emission_initializer)
    self.init_initializer = initializers.get(init_initializer)
    self.state_size = [1, self.n, 1]
    self.output_size = self.n

  def build(self, input_shape):
    self.emission_kernel = self.add_weight(
        shape=(self.n, input_shape[-1]),
        initializer=self.emission_initializer,
        name='emission_kernel') # closely related to B
    self.transition_kernel = self.add_weight(
        shape=(self.n, self.n),
        initializer=self.transition_initializer,
        name='transition_kernel') # closely related to A
    self.init_kernel = self.add_weight(
        shape=(self.n),
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
    
    E = tf.linalg.matvec(self.B, inputs, transpose_a=False, name="E")
    forward = tf.multiply(E, R, name="forward")
    S = tf.reduce_sum(forward, axis=-1, name="loglik")
    loglik = old_loglik + tf.math.log(S)
    forward = forward / tf.expand_dims(S, -1)
    is_init = tf.zeros(batch_size, dtype='int8', name="is_init")
    new_state = [is_init, forward, loglik]
    new_state = [new_state] if nest.is_nested(states) else new_state
    if verbose:
        print ("new_state", new_state)
    return new_state, new_state

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """ initially the HMM starts in state 0 """
    is_init = tf.ones(batch_size, dtype='int8')
    forward = tf.zeros([batch_size, self.n], dtype=np.float32)
    loglik = tf.zeros(batch_size)
    S = [is_init, forward, loglik]
    return S
 
  def get_config(self):
    config = {
        'n': self.units,
        'transition_initializer':
            initializers.serialize(self.transition_initializer),
        'emission_initializer':
            initializers.serialize(self.emission_initializer),
         'init_initializer':
            initializers.serialize(self.init_initializer),
    }
    config.update(_config_for_enable_caching_device(self))
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

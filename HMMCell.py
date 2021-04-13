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
    self.state_size = self.n + 1
    self.output_size = self.n + 1

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
        name='init_kernel') # closely related to initial distribution of 0-th hidden state
    
    self.built = True

  def call(self, inputs, states, training=None):
    prev_output = states[0] if nest.is_nested(states) else states
    # convert parameter matrices to stochastic matrices for transition (A) and emission probs (B)
    # TODO: this could be more efficient, maybe using tensorflow.python.keras.constraints?
    I = tf.nn.softmax(self.init_kernel, axis=-1, name="I")
    A = tf.nn.softmax(self.transition_kernel, axis=-1, name="A")
    B = tf.nn.softmax(self.emission_kernel, axis=-1, name="B")
    print ("I=\n", I, "\nA=\n", A, "\nB=\n", B)
    print ("prev_output=\n", prev_output, " shape", prev_output.shape)
    
    R0 = prev_output[0,0] * I # only in first time step
    print (f"R0:{R0}")
    R1 = tf.linalg.matvec(A, prev_output[0,1:], transpose_a=True)
    print (f"R1:{R1}")
    R = R0+R1
    print (f"R:{R}")
    E = tf.linalg.matvec(B, inputs, transpose_a=False, name="E")
    output = tf.multiply(E, R)
    print (f"output:{output}")
    output = [tf.concat([np.array([0.], dtype=np.float32), output[0]], axis=0)]
    new_state = [output] if nest.is_nested(states) else output
    return output, new_state

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """ initially the HMM starts in state 0 """
    I = np.zeros(self.n+1, dtype=np.float32)
    I[0] = 1.0
    I = np.tile(I, (batch_size,1))
    return I

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
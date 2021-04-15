"""Generate simulated data from the occasionally dishonest casino.
The HMM has 
 * state space {0,1} = {F,L}
 * emission alphabet {1,..., 6}
 * transition distribution given by matrix A
 * emission distribution given by matrix B
 * always starts in state F
"""

import tensorflow as tf
import numpy as np

sigma = [0,1,2,3,4,5]
s = len(sigma)  # emission alphabet 0,..., 5
A = np.array([[19., 1], [3, 17]]) / 20.0
B = np.array([[10.,10,10,10,10,10],[6,6,6,6,6,30]]) / 60.0
n = 2 # number of states
T = 12 # sequence length

def casino_generator():
    y = np.zeros((T, s), dtype=np.float32)
    q = None
    for i in range(T):
        if q is None:
            q = 0
        else:
            q = np.random.choice(range(n), p=A[q])
        c = np.random.choice(sigma, p=B[q])
        y[i,c] = 1.0
                                 
    yield y

def get_casino_dataset():
    dataset = tf.data.Dataset.from_generator(
         casino_generator,
         output_signature=(tf.TensorSpec(shape=(None, s), dtype=tf.float32)))
    return dataset

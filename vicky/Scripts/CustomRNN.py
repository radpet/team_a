# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import tensorflow as tf

# <codecell>

class CustomRNN(tf.contrib.rnn.GRUCell):
    def __init__(self, *args, **kwargs):
        returns = super(CustomRNN, self).__init__(*args, **kwargs) # create an lstm cell
        return returns
    def __call__(self, inputs, state):
        output, next_state = super(CustomRNN, self).__call__(inputs, state)
        return next_state, next_state # return two copies of the state, instead of the output and the state

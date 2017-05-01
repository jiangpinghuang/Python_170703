from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class AttentionGRUCell(RNNCell):
    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "attention_gru_cell"):
            with vs.variable_scope("gates"):
                if inputs.get_shape()[-1] != self._num_units + 1:
                    raise ValueError("Input should be passed as word input concatenated with 1D attention on end axis")
                inputs, g = array_ops.split(inputs,
                        num_or_size_splits=[self._num_units, 1],
                        axis=1)
                r = _linear([inputs, state], self._num_units, True)
                r = sigmoid(r)
            with vs.variable_scope("candidate"):
                r = r * _linear(state, self._num_units, False)
            with vs.variable_scope("input"):
                x = _linear(inputs, self._num_units, True)
            h_hat = self._activation(r + x)
            new_h = (1 - g) * state + g * h_hat
        return new_h, new_h

def _linear(args, output_size, bias, bias_start=0.0):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value
    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            "weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                        "biases", [output_size],
                      dtype=dtype,
                    initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
        return nn_ops.bias_add(res, biases)

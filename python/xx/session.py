from __future__ import absolute_import

import numpy as np
from . import autodiff

_all_variable_inits = []


def global_variables_initializer():
    global _all_variable_inits
    init = autodiff.init_op(_all_variable_inits)
    _all_variable_inits = []
    return init

class Session(object):
    """Executor computes values for given set of nodes in computation graph."""
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        return None
    def run(self, fetch, feed_dict=None):
        feed_dict = feed_dict if feed_dict else {}
        if not isinstance(fetch, list):
            fetch = [fetch]
        for n, val in feed_dict.items():
            if not isinstance(val, np.ndarray):
                if not isinstance(val, list):
                    val = [val]
                feed_dict[n] = np.array(val)
        exe = autodiff.Executor(fetch)
        ans = exe.run(feed_dict)
        for node, val in ans.items():
            if val.shape == (1,):
                ans[node] = val[0]
        if len(fetch) == 1:
            return ans[fetch[0]]
        return [ans[node] for node in fetch]


def placeholder(dtype=None, shape=None):
    placeholder_node = autodiff.placeholder_op(dtype)
    return placeholder_node


def Variable(init=None, dtype=None):
    variable_node = autodiff.variable_op()
    if init is not None:
        if isinstance(init, autodiff.Node):
            value_node = init
        else:
            if not isinstance(init, np.ndarray):
                if not isinstance(init, list):
                    init = [init]
                    init = np.array(init)
            value_node = autodiff.const_op(init)
        _all_variable_inits.append(autodiff.assign_op(variable_node, value_node))
    return variable_node

def assign(node, c):
    assign_node = autodiff.assign_op(node, c)
    return assign_node

def constant(c, shape=None):
    if shape:
        shape = tuple(x for x in shape)
        c = np.broadcast_to(c, shape)
    const_node = autodiff.const_op(c)
    return const_node

def reduce_sum(node, reduction_indices=[0]):
    if not isinstance(reduction_indices, list):
        reduction_indices = [reduction_indices]
    reduce_sum_node = autodiff.reducesum_op(node, reduction_indices)
    return reduce_sum_node

def reduce_mean(node, reduction_indices=[0]):
    reduce_mean_node = autodiff.reducesum_op(node, reduction_indices) / autodiff.size_op(node, reduction_indices)
    return reduce_mean_node

def log(node):
    log_node = autodiff.log_op(node)
    return log_node

def matmul(node_A, node_B):
    matmul_node = autodiff.matmul_op(node_A, node_B)
    return matmul_node

def equal(node_A, node_B):
    equal_node = autodiff.equal_op(node_A, node_B)
    return equal_node

def argmax(node, axis=None):
    argmax_node = autodiff.argmax_op(node, axis)
    return argmax_node

def cast(node, dtype):
    cast_node = autodiff.cast_op(node, dtype)
    return cast_node

def random_normal(shape=None, loc=0.0, stddev=1.0):
    return np.random.normal(loc=loc, scale=stddev, size=shape)

def reshape(node, shape):
    shape = tuple(x for x in shape)
    reshape_node = autodiff.reshape_op(node, shape)
    return reshape_node
from . import autodiff
from . import session

def softmax(node):
    softmax_node = autodiff.softmax_op(node)
    return softmax_node

def relu(node):
    relu_node = autodiff.relu_op(node)
    return relu_node

def softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=[1], name=None):
    return -autodiff.reducesum_op(labels * autodiff.log_op(autodiff.softmax_op(logits)), reduction_indices=dim)

def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME'):
    conv2d_node = autodiff.conv2d_op(input, filter, strides, padding)
    return conv2d_node

def max_pool(input, ksize, strides=[1, 1, 1, 1], padding='SAME'):
    maxpool_node = autodiff.maxpool_op(input, ksize, strides, padding)
    return maxpool_node

def dropout(x, keep_prob):
    dropout_node = autodiff.dropout_op(x, keep_prob)
    return dropout_node
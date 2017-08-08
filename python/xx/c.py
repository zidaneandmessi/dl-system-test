import numpy as np
import ctypes
import os

cdll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "cpp.so"))

def conv2d(input, filter, output):
    filter_pointer = filter.ctypes.data_as(POINTER(c_float))
    output_pointer = output.ctypes.data_as(POINTER(c_float))
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    filter_height = filter.shape[0]
    filter_width = filter.shape[1]
    out_height = output.shape[1]
    out_width = output.shape[2]
    in_channels = input.shape[3]
    out_channels = filter.shape[3]
    output[:] = np.zeros(output.shape)
    if mode == 'valid':
        input_pointer = input.ctypes.data_as(POINTER(c_float))
    elif mode == 'same':
        in_height = in_height + filter_height - 1
        in_width = in_width + filter_width - 1
        padding = (in_height - filter_height) / 2
        input_matrix = np.zeros((in_height, in_width))
        input_matrix[padding:-padding, padding:-padding] = input
        input_pointer = input_matrix.ctypes.data_as(POINTER(c_float))
    cdll.conv2d(input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels)

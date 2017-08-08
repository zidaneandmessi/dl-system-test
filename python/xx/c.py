import numpy as np
import ctypes
import os

cdll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "cpp.so"))

def conv2d(input, filter, output):
    input_pointer = input.ctypes.data_as(POINTER(c_float))
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
    if mode == 'valid':
        padding = 0
    elif mode == 'same':
        padding = 1
    cdll.conv2d(input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels, padding)

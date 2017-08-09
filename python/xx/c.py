import numpy as np
import ctypes
import os

cdll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "cpp.so"))

def conv2d(input, filter, output):
    output[:] = np.zeros(output.shape)
    filter_pointer = filter.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    filter_height = filter.shape[0]
    filter_width = filter.shape[1]
    out_height = output.shape[1]
    out_width = output.shape[2]
    in_channels = input.shape[3]
    out_channels = filter.shape[3]
    in_height = in_height + filter_height - 1
    in_width = in_width + filter_width - 1
    padding = (in_height - out_height) / 2
    input_matrix = np.zeros((batch, in_height, in_width, in_channels))
    for i in range(batch):
        for k in range(in_channels):
            input_matrix[i, padding:-padding, padding:-padding, k] = input[i, :, :, k]
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.correlate2d(input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels)

def conv2dgrad1(input, filter, output):
    output[:] = np.zeros(output.shape)
    filter_pointer = filter.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    filter_height = filter.shape[0]
    filter_width = filter.shape[1]
    out_height = output.shape[1]
    out_width = output.shape[2]
    in_channels = output.shape[3]
    out_channels = filter.shape[3]
    in_height = in_height + filter_height - 1
    in_width = in_width + filter_width - 1
    padding = (in_height - out_height) / 2
    input_matrix = np.zeros((batch, in_height, in_width, out_channels))
    for i in range(batch):
        for k in range(out_channels):
            input_matrix[i, padding:-padding, padding:-padding, k] = input[i, :, :, k]
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.correlate2dgrad1(input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels)

def conv2dgrad2(input, filter, output):
    output[:] = np.zeros(output.shape)
    filter_pointer = filter.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    filter_height = filter.shape[1]
    filter_width = filter.shape[2]
    out_height = output.shape[0]
    out_width = output.shape[1]
    in_channels = input.shape[3]
    out_channels = filter.shape[3]
    in_height = in_height - 1 + out_height
    in_width = in_width - 1 + out_width
    input_matrix = np.zeros((batch, in_height, in_width, in_channels))
    padding = (out_height - 1) / 2
    for i in range(batch):
        for k in range(in_channels):
            input_matrix[i, padding:-padding, padding:-padding, k] = input[i, :, :, k]
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.correlate2dgrad2(input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels)
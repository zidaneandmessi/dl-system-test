import numpy as np
import ctypes
import os

cdll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "cpp.so"))

#profile
def conv2d(input, filter, output):
    output[:] = np.zeros(output.shape)
    origin_input_pointer = input.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
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
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.correlate2d(input_pointer, origin_input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels, padding)

#profile
def conv2dgrad1(input, filter, output):
    output[:] = np.zeros(output.shape)
    origin_input_pointer = input.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
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
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.correlate2dgrad1(input_pointer, origin_input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels, padding)

def conv2dgrad2(input, filter, output):
    output[:] = np.zeros(output.shape)
    origin_input_pointer = input.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
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
    padding = (out_height - 1) / 2
    input_matrix = np.zeros((batch, in_height, in_width, in_channels))
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.correlate2dgrad2(input_pointer, origin_input_pointer, filter_pointer, output_pointer, batch, in_height, in_width, filter_height, filter_width, out_height, out_width, in_channels, out_channels, padding)

#profile
def maxpool(input, ksize, stride, output):
    output[:] = np.zeros(output.shape)
    origin_input_pointer = input.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    pool_height = ksize[1]
    pool_width = ksize[2]
    out_height = output.shape[1]
    out_width = output.shape[2]
    in_channels = input.shape[3]
    padding = ((in_height / stride - 1) * stride + ksize[1] - in_height) / 2
    in_height = in_height + padding * stride
    in_width = in_width + padding * stride
    input_matrix = np.zeros((batch, in_height, in_width, in_channels))
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.maxpool(input_pointer, origin_input_pointer, output_pointer, batch, in_height, in_width, pool_height, pool_width, out_height, out_width, in_channels, stride, padding)
    
#profile
def maxpoolgrad(input, gradient, ksize, stride, output):
    output[:] = np.zeros(output.shape)
    origin_input_pointer = input.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    gradient_pointer = gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    grad_height = gradient.shape[1]
    grad_width = gradient.shape[2]
    pool_height = ksize[1]
    pool_width = ksize[2]
    out_height = output.shape[1]
    out_width = output.shape[2]
    in_channels = input.shape[3]
    padding = ((in_height / stride - 1) * stride + ksize[1] - in_height) / 2
    in_height = in_height + padding * stride
    in_width = in_width + padding * stride
    input_matrix = np.empty([batch, in_height, in_width, in_channels])
    input_pointer = input_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ans = np.zeros((batch, in_height, in_width, in_channels))
    ans_pointer = ans.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cdll.maxpoolgrad(input_pointer, gradient_pointer, origin_input_pointer, ans_pointer, output_pointer, batch, in_height, in_width, grad_height, grad_width, pool_height, pool_width, out_height, out_width, in_channels, stride, padding)
    # batch = input.shape[0]
        # in_height = input.shape[1]
        # in_width = input.shape[2]
        # in_channels = input.shape[3]
        # for i in range(batch):
        #     for k in range(in_channels):
        #         padding = ((in_height / strides[1] - 1) * strides[1] + ksize[1] - in_height) / 2
        #         input_matrix = np.zeros((in_height + padding * 2, in_width + padding * 2))
        #         if padding == 0:
        #             input_matrix[:, :] = input[i, :, :, k]
        #             output_val[i, :, :, k] = pooling_grad(input_matrix, ksize, strides[1], gradient[i, :, :, k])
        #         else:
        #             input_matrix[padding:-padding, padding:-padding] = input[i, :, :, k]
        #             output_val[i, :, :, k] = pooling_grad(input_matrix, ksize, strides[1], gradient[i, :, :, k])[padding:-padding, padding:-padding]
        
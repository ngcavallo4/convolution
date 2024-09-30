import random 
import numpy as np

def apply_func(func: callable, vec: list, n):

    output = np.zeros_like(vec)

    for index, t in enumerate(vec):
        output[index] = func(t, n)

    return output

def discrete_convolution(f: list, g: list):

    conv_vec = np.convolve(f,g)

    return conv_vec


# def discrete_convolution(f: list,g: list):

#     len_f = len(f)
#     len_g = len(g)

#     conv_length = len_f + len_g - 1

#     conv_vec = np.zeros(conv_length)

#     for t in range(conv_length):
#         for k in range(len_f):
#             if 0 <= t-k < len_g:
#                 n1 = f[k]
#                 n2 = g[t-k]
#                 conv_vec[t] += n1*n2

#     return conv_vec

def count_zeros_ones(vec):

    vec = np.array(vec)
    vec_zeros = np.count_nonzero(vec==0.0)
    vec_ones = np.count_nonzero(vec==1.0)
    vec_neg = np.count_nonzero(vec==-1.0)

    return vec_zeros, vec_ones, vec_neg
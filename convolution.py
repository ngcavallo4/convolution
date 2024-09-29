import random 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

# def apply_func(func: callable, vec: list, n):

#     output = np.zeros_like(vec)

#     for index, t in enumerate(vec):
#         output[index] = func(t, n)

#     return output

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

def apply_func(func: callable, vec: list, n: int):
    output = np.zeros_like(vec)

    # Parallelizing using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda i: func(vec[i], n), range(len(vec))))
    
    output[:] = results
    return output

def compute_convolution_element_wrapper(args):
    """A wrapper function to unpack arguments for compute_convolution_element"""
    t, f, g, len_f, len_g = args
    return compute_convolution_element(t, f, g, len_f, len_g)

def compute_convolution_element(t, f, g, len_f, len_g):
    conv_value = 0
    for k in range(len_f):
        if 0 <= t - k < len_g:
            conv_value += f[k] * g[t - k]
    return conv_value

def discrete_convolution(f: list, g: list):
    len_f = len(f)
    len_g = len(g)
    conv_length = len_f + len_g - 1

    # Preparing arguments for the parallel function
    args = [(t, f, g, len_f, len_g) for t in range(conv_length)]

    with ProcessPoolExecutor() as executor:
        conv_vec = list(executor.map(compute_convolution_element_wrapper, args))

    return np.array(conv_vec)
import numpy as np
import matplotlib.pyplot as plt
import convolution as cv

## Part b 
def f(t,n):
    if t == 20:
        return 2.0
    else:
        return 0.0

def g(t,n):
    if t == 5:
        return 3.0
    else:
        return 0.0
    
def f2(t,n):
    if t == 20:
        return 1.0
    elif t == 5:
        return 1.0
    else:
        return 0.0

## Part c

def g2(t,n):
    if t>10:
        return 0.0
    elif t < 10 or t >= 0:
        return 1/10

## Part d

def s(t, n):
    if t > (2*n)/3:
        return 0
    elif t > n/3:
        return 1
    else:
        return 1

def r(t, n):

    rand_num = np.random.randint(-1,1)
    return s(t,n) + rand_num

## Part e

def g3(t,m):

    if t > m:
        return 0
    elif t>= 0:
        return 1/m
    
def part_b_1(n):
    vec_f = np.arange(0,n,dtype=float)
    f_stem = cv.apply_func(f, vec_f, n)

    vec_g = np.arange(0,n,dtype=float)
    g_stem = cv.apply_func(g, vec_g, n)

    h_stem_mine = cv.discrete_convolution(f_stem,g_stem)
    vec_h = np.arange(0,len(h_stem_mine))

    plt.figure()

    f_handle = plt.stem(vec_f, f_stem, markerfmt='bo', linefmt='b-', label='f')
    g_handle = plt.stem(vec_g, g_stem, markerfmt='ro', linefmt='r-', label='g')
    h_handle = plt.stem(vec_h, h_stem_mine, markerfmt='go', linefmt='g-', label='h')

    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title("f ⊗ g")
    plt.legend(loc='upper right')

    # ax2.set_xlabel("t")
    # ax2.set_ylabel("Amplitude")
    # ax2.set_title("g(t)")
    # ax2.legend(loc='upper right')

    # ax3.set_xlabel("t")
    # ax3.set_ylabel("Amplitude")
    # ax3.set_title("h(t)")
    # ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def part_b_2(n):
    vec_f = np.arange(0,n,dtype=float)
    f_stem = cv.apply_func(f2, vec_f, n)

    vec_g = np.arange(0,n,dtype=float)
    g_stem = cv.apply_func(f2, vec_g, n)

    h_stem_mine = cv.discrete_convolution(f_stem,g_stem)
    vec_h = np.arange(0,len(h_stem_mine))

    plt.figure()

    f_handle = plt.stem(vec_f, f_stem, markerfmt='bo', linefmt='b-', label='f')
    g_handle = plt.stem(vec_g, g_stem, markerfmt='bo', linefmt='b-', label='f')
    h_handle = plt.stem(vec_h, h_stem_mine, markerfmt='go', linefmt='g-', label='f ⊗ f')

    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title("f ⊗ f")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def part_e(n,m):
    vec_s = np.arange(0,n, dtype=float)
    s_stem = cv.apply_func(s,vec_s, n)
    s_stem_left = s_stem[:6667]
    s_stem_right = s_stem[6667:]
    s_zeros_left, s_ones_left, x = cv.count_zeros_ones(s_stem_left)
    s_zeros_right, s_ones_right, y = cv.count_zeros_ones(s_stem_right)
    print(f"{s_zeros_left} zeros < 6667 in s(t), {s_zeros_right} > 6667, \n {s_ones_left} ones < 6667 in s(t), {s_ones_right} > 6667 \n")

    vec_r = np.arange(0,n,dtype=float)
    r_stem = cv.apply_func(r, vec_r, n)
    r_stem_left = r_stem[:6667]
    r_stem_right = r_stem[6667:]
    r_zeros_left, r_ones_left, z = cv.count_zeros_ones(r_stem_left)
    r_zeros_right, r_ones_right, r_neg_right = cv.count_zeros_ones(r_stem_right)
    print(f"{r_zeros_left} zeros < 6667 in r(t), {r_zeros_right} > 6667, \n {r_ones_left} ones < 6667 in s(t), {r_neg_right} -1s > 6667\n ")

    vec_g = np.arange(0,n,dtype=float)
    g_stem = cv.apply_func(g3, vec_g, m)

    # r ⊗ g
    h_stem = cv.discrete_convolution(r_stem,g_stem)
    vec_h = np.arange(0,len(h_stem))

    # # r ⊗ g ⊗ g
    h_stem = cv.discrete_convolution(h_stem,g_stem)
    vec_h = np.arange(0,len(h_stem))

    # # r ⊗ g ⊗ g ⊗ g
    h_stem = cv.discrete_convolution(h_stem,g_stem)
    vec_h = np.arange(0,len(h_stem))

    fig, axs = plt.subplots(1,2, figsize = (14,6))

    # ax1 = axs[0,0]
    # ax2 = axs[0,1]
    ax3 = axs[0]
    ax4 = axs[1]

    # s_handle = ax1.stem(vec_s, s_stem, markerfmt='bo', linefmt='b-', label='s')
    # r_handle = ax2.stem(vec_r, r_stem, markerfmt='bo', linefmt='b-', label='r')
    g_handle = ax3.stem(vec_g, g_stem, markerfmt='bo', linefmt='b-', label='g')
    h_handle = ax4.stem(vec_h, h_stem, markerfmt='bo', linefmt='b-', label='h ⊗ g ⊗ g')

    # ax1.set_xlabel("t")
    # ax1.set_ylabel("Amplitude")
    # ax1.set_title("s(t)")
    # ax1.legend(loc='upper right')

    # ax2.set_xlabel("t")
    # ax2.set_ylabel("Amplitude")
    # ax2.set_title("r(t)")
    # ax2.legend(loc='upper right')

    ax3.set_xlabel("t")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("g(t)")
    ax3.legend(loc='upper right')

    ax4.set_xlabel("t")
    ax4.set_ylabel("Amplitude")
    ax4.set_title("r ⊗ g")
    ax4.legend(loc='upper right')

    fig.suptitle(f'N = {n}, M = {m}')
    plt.tight_layout()
    plt.show()

part_b_1(50)
part_b_2(50)
# part_e(10000,1000)
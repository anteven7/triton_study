import triton
import triton.language as tl
import torch

# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by hardware 
# vendors themselves as part of so-called “kernel libraries” (e.g., cuBLAS). Unfortunately, these
# libraries are often proprietary and cannot be easily customized to accommodate the needs of modern 
# deep learning workloads (e.g., fused activation functions). In this tutorial, you will learn how 
# to implement efficient matrix multiplications by yourself with Triton, in a way that is easy to
#  customize and extend.

# Let's check out how a naive matmul would be in python code.
# This is a naive (and super slow) implementation.
#
def naive_matmul(a, b):
# a: (M, K), b: (K, N)
    M, K = a.shape
    _, N = b.shape

    # 1. Initialize result matrix of shape (M, N)
    C = torch.zeros((M, N))

    # 2. Loop over rows of A
    for i in range(M):
    # 3. Loop over columns of B
        for j in range(N):
            # 4. Loop over the 'shared' dimension K to calculate the sum
            acc = 0
            for k in range(K):
                acc += a[i, k] * b[k, j]
            
        C[i, j] = acc

    return C


# lets see how the for logic will translate to triton's 
# main goal here is to block-multiply to feed the tensor cores as much as possible.
#
#
# we choose now all the chunck of rows we want to compute:
# for m in range(0, M, BLOCK_SIZE_M): this is a pararell execution for each block size, 
#
# therefore, we choose now all the chuncks of colums we want to compute:
#   for n in range(0, N, BLOCK_SIZE_N): this is a pararell execution for each block size 
#
#   having both of this chunks, we calculate the dot product between them 
#     acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#     for k in range(0, K, BLOCK_SIZE_K):
#       a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#       b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#       acc += dot(a, b)
#     C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc

# lets illustrate it:
#
# A = 
#         [0.12, 0.85, 0.34, 0.22, 0.57, 0.91, 0.14, 0.63],
#         [0.61, 0.29, 0.73, 0.51, 0.08, 0.42, 0.88, 0.19],
#         [0.47, 0.53, 0.11, 0.64, 0.82, 0.37, 0.25, 0.71],
#         [0.92, 0.04, 0.86, 0.39, 0.15, 0.68, 0.54, 0.31],
#         [0.38, 0.76, 0.59, 0.27, 0.94, 0.12, 0.49, 0.83],
#         [0.74, 0.19, 0.43, 0.81, 0.36, 0.70, 0.67, 0.28],
#         [0.55, 0.68, 0.21, 0.93, 0.48, 0.16, 0.79, 0.35],
#         [0.26, 0.89, 0.75, 0.14, 0.63, 0.52, 0.41, 0.97]
#
#
#
# and we can tile it this way:
#
# [a, b]
# [c, d], being a = [0.12, 0.85, 0.34, 0.22, 
#                   [0.61, 0.29, 0.73, 0.51,
#                   [0.47, 0.53, 0.11, 0.64,
#                   [0.92, 0.04, 0.86, 0.39,
#
#
# therefore, we can assume a matrix B as the same as A with different numbers. We have then:
#
# A = [a, b]    
#     [c, d]    
#                      Where A and B are shapes (8, 8), being their multiplication the shape 
#                      (M, K) x (K, N) in this case being M, K, N = 8.
#
#                      The individual letters are shapes (4,4), or more
#                      precisely (BLOCK SIZE M, BLOCK SIZE K) in the case of A and 
#                      (BLOCK SIZE K, BLOCK SIZE N) in the case of B
# B = [e, f]
#     [g, h]
#
#
# Therefore, when multyplying them:
##               [a, b] x [e]    , as [a, b] is the shape (BLOCK SIZE M, K) 
#                         [g]      and [e] is the shape (K, BLOCK_SIZE_N)
#                                      [g] 
# would exactly mean:
#
#   for m in range(0, M, BLOCK_SIZE_M):
#       for n in range(0, N, BLOCK_SIZE_N):
#
#           a = A[m:m+BLOCK_SIZE_M, 0:K]
#           b = B[0:K,n:n+BLOCK_SIZE_N']
#           
#           c = dot(a,b)
#
# This, indeed, is a form of calculating the dot product of two matrix, but we will tile it even further 
# as we use the BLOCK SIZE K distribute more the multiplication of matrix [a, b] and [e].  
#                                                                                    [g]
#
#for k in range(0, K, BLOCK_SIZE_K):
#       a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#       b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#
# What this line does is that in the case of the matrix [a, b] and [e] (assuming that we have BLOCK_SIZE_K = 4 and it 0):
#                                                                  [g]
#
#       a = a (shape(BLOCK_SIZE_M, BLOCK_SIZE_K)), memory offset -> A[0:BLOCK_SIZE_M, 0:BLOCK_SIZE_K]
#       b = e (shape(BLOCK_SIZE_K, BLOCK_SIZE_N)), memory offset -> B[0:BLOCK_SIZE_K, 0:BLOCK_SIZE_N]
#       acc += dot(a,b)
#
#
# Then it 2:
#
#       a = b (shape(BLOCK_SIZE_M, BLOCK_SIZE_K)), memory offset -> A[0:BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]
#       b = g (shape(BLOCK_SIZE_K, BLOCK_SIZE_N)), memory offset -> B[k:k+BLOCK_SIZE_K, 0:BLOCK_SIZE_N]
#
#       acc += dot(a,b)
#
#
# This will be done for [a, b] and the following colum of [f],
#                                                         [h]
#
#
# Therefore:
#
# [a,b] [e, f]      [ae + bg, af + bh]
# [c,d] [g, h]  ->  [ce + dg, cf + dh] 
#
# which, if we look at the accumulations we were doing before (dot(a,e) and dot(b,g)) we arrive 
# at the same destination.
#

#l2 cache reuse, triton-lang picture.

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# import os
# os.environ['TRITON_INTERPRETER']="1"

# let's let triton figure out what are the best BLOCK_SIZE_M BLOCK_SIZE_N.

autotune_conf = [
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                num_warps=8),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                num_warps=2),
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                num_warps=2),
# Good config for fp8 inputs.
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                num_warps=8),
triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                num_warps=8),
triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4),
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                num_warps=4)
]

@triton.autotune(configs = autotune_conf, key=['M', 'N', 'K'])
# to continue with the explanation we need to analyze the concept of L2 cache optimization.
# when doing a naive matmul with tiling like the example before we do it in a so called row-major
# ordering. This means that each pid computes something like this:

# Lets say A is a matrix of 15 tiles     

#     [0,   1,  2,  3]
#     [4,   5,  6,  7]
#     [8,   9, 10, 11]
#     [12, 13, 14, 15]


# and target matrix C as this:

# [a, b, c, d]
# [e, f, g, h]
# [i, j, k, l]
# [m, n, o, p]


# When you separate this into pids for calculations, it ends up being something like:

#     PID = 0, calculates chunk a
#     [x, x, x, x]        [x, _, _, _]
#     [_, _, _, _]        [x, _, _, _]
#     [_, _, _, _]        [x, _, _, _]
#     [_, _, _, _]        [x, _, _, _]
#     PID = 1, calculates chunk b
#     [x, x, x, x]        [_, x, _, _]
#     [_, _, _, _]        [_, x, _, _]
#     [_, _, _, _]        [_, x, _, _]
#     [_, _, _, _]        [_, x, _, _]
#     PID = 2 ...
#     [x, x, x, x]        [_, _, x, _]
#     [_, _, _, _]        [_, _, x, _]
#     [_, _, _, _]        [_, _, x, _]
#     [_, _, _, _]        [_, _, x, _]
#     PID = 3 ...
#     [x, x, x, x]        [_, _, _, x]
#     [_, _, _, _]        [_, _, _, x]
#     [_, _, _, _]        [_, _, _, x]
#     [_, _, _, _]        [_, _, _, x]

# obviously, each chunk x is iterated and accumulated for the calculation of the result chunk.

# If we look at the number of rows and cols of chunk loaded, we see we load 1 row of A and 4 cols 
# of B. In total thats around 5 rows/cols of chunks. 

# Luckily there is a smarter implementation. Lets look at pids 4 and 5:

#     PID = 4
#     [_, _, _, _]        [x, _, _, _]
#     [x, x, x, x]        [x, _, _, _]
#     [_, _, _, _]        [x, _, _, _]
#     [_, _, _, _]        [x, _, _, _]
#     PID = 5
#     [_, _, _, _]        [_, x, _, _]
#     [x, x, x, x]        [_, x, _, _]
#     [_, _, _, _]        [_, x, _, _]
#     [_, _, _, _]        [_, x, _, _]

# As you can appreciate, if we were to replace the pids 2 and 3 for 4 and 5 we would only have to load
# 4 cols/rows of chunks, saving us 1!

# this is used to reduce the amount of loading in the L2 cache, which significantly speeds up the 
# whole kernel with bigger sizes. This is called group-major ordering.

# the relocation is somehow more difficult, as triton loads blocks into SMs based on the order of 
# PIDs, meaning that if we force 4 and 5 to be in the same SM, we would probably to have all 0, 1, 2,
# 3, 4, and 5 PIDs.

# Then, as a result, the only option would be to reorder the matrix so we target the chunks we want by
# moving the PIDs index, getting as a result something like:


#     [0,  2,  4,  6]
#     [1,  3,  5,  7]
#     [8, 10, 12, 14]
#     [9, 11, 13, 15] 

# Now, 0 through 3 correspond to group-major ordering! Notice in this example we can visualize it as splitting our
# PIDs into "groups" demarcated by the dashed lines.

#     [0,  2,   4,  6]
#     [1,  3,   5,  7]
#     ----------------
#     [8, 10,  12, 14]
#     [9, 11,  13, 15] 


# then, one question could arise: why are we then permutating the numbers if both stacks are going to 
# the same SM?

# well, the response is the scheduler of triton. It alwasy goes sequential, as it doesnt understand 2D:
# it computes pid 0, pid 1, pid 2.... so even if they are getting the same work, the order in which they 
# compute the chunks dictates which remains in the SRAM.


@triton.jit 
def matmul_kernel(
    # pointers to the matrices
    a, b, c_ptr,
    # dimensions of matrices
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    # now the constants, we could not pass it bc the autotune.
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
    ACTIVATION: tl.constexpr):

    # our first duty is to calculate the pids, map them to the block of 
    # C they should compute, following the goup-major order to L2 cache reuse.a

    pid = tl.program_id(0) # as always calculating the pid im In
    # now lets see how many pids we have in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # after we know how many pids, we proceed to calculate the number of pids in the group
    # the groups are made splitting through M, as we saw before. 

    num_pid_in_group = GROUP_SIZE_M * num_pid_n # this is the n of pids per group

    # having the number of pids in group, we can then identify each group
    # pid being the one im at divided by number of pids in group
    # for example, pid 7 and num_pid_in_group 8 would mean 7//8 = 0 (group 0)

    group_id = pid // num_pid_in_group

    # lets locate now the first pid in each group
    # if we are in group 1 and group size is 2, we get pid 2 as the first of the 2 group 

    first_pid_m = group_id * GROUP_SIZE_M    

    # lets now calculate the last group size in case NUM PID M is not divisible by GROUP SIZE M 

    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # now we calculate PID coordinates. The cool part:

    #     Col 0  Col 1  Col 2  Col 3
    #        ---------------------------
    # Row 0 |   0      2      4      6  |
    # Row 1 |   1      3      5      7  | -> GROUP 0 (Starts at Row 0)
    #       |---------------------------|
    # Row 2 |   8     10     12     14  |
    # Row 3 |   9     11     13     15  | -> GROUP 1 (Starts at Row 2)


    # pid_m aims to calculate the row

    # first pid m = the first pid on the group
    # pid = current pid
    # num pid in group = n pids per group = 8
    # group size m = the safe group size = 2 

    # pid % num_pid_in_group finds the relative id, if we are in pid 11 and num pid in group is 8, we get 3. 
    # This means PID 11 is the 3rd item (starting from 0) inside Group 1. By stripping away the global PID number, 
    # both formulas can now figure out where this tile belongs relative to the start of its own group. 

    # this step is crucial, the relative_id % group_size_m is what ccreates the zigzag:
    # relative position 0 % 2 = 0 (row 0), relative position 1 % 2 = row 1
    # relative position 2 mod 2 = row 0, relative position 3 % 2 = 1 = row 1
    # this reshapes the order of pids within the rows of the group
    # we sum first_pid_m just represents the starting row, so we add it up

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)

    # pid_n aims to calculate the the colum

    # pid % num_pid_in_group as we said calculates the relative pid within the group, in this case 3 
    # we do floor division against the group height -> 3//2 = 1 

    # Why this works: Because our group is 2 rows tall, the GPU spends exactly 2 sequential PIDs in every 
    # column before moving to the right.

    # Relative IDs 0 and 1 belong to Col 0 (0//2 = 0, 1//2 = 0)
    # Relative IDs 2 and 3 belong to Col 1 (2//2 = 1, 3//2 = 1)
    # Relative IDs 4 and 5 belong to Col 2 (4//2 = 2, 5//2 = 2)

    pid_n = (pid % num_pid_in_group) // group_size_m


    # now that we have the ids of the pid we are going to compute, we calculate offsets

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M 
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N    
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # offsets are a list of the "index" of the rows and colums we are going to compute.  
    # we then calculate the pointers by taking those numbers and the strides to precisely get the memory addreses

    a_ptrs = a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # now we have our matrix a (blocksizem, blocksizek) and b (blocksizek, blocksizen) and its pointers from the corresponding
    # pid we wanted to target.

    # the mission now is to iterate towards the K dimension (as it would be super long, we iterate blocksizek by blocksizek), as the a and b
    # pointers have that dim.

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        # we load the block of rows and colums we are going to compute to resolve the pid
        # mask means that 

        a_block = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_block = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # tl dot makes 
        accumulator = tl.dot(a_block, b_block, accumulator)

        # and we advance pointers manually for next it
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)

# ======================================================================================
# Why do we perform dequantization on-the-fly inside the kernel before matmul?
# In LLM inference, weight-only quantization (W8A16) aims to overcome the "Memory Wall"
# bottleneck, where execution speed is limited by VRAM memory bandwidth rather than GPU
# compute capacity. By keeping the weights in 8-bit precision (INT8) in global memory (DRAM),
# we cut the weight data traffic over the memory bus in half. Inside the GPU kernel, we
# load these INT8 weights into SRAM (fast local cache) and then convert (dequantize) them
# to 16-bit float (FP16) on-the-fly right before performing the tensor-core matrix multiplication
# (tl.dot). This is extremely efficient because register-level/SRAM math is fast, and we
# completely avoid the overhead of loading larger FP16 weights from slow VRAM.
# ======================================================================================

@triton.autotune(configs = autotune_conf, key=['M', 'N', 'K'])
@triton.jit
def matmul_on_the_fly_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n

    scales = tl.load(scales_ptrs, mask=offs_bn < N, other=1.0).to(tl.float16)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_block = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_block = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Cast INT8 weights to FP16
        b_block_fp16 = b_block.to(tl.float16)

        # Dequantize weights on-the-fly: B = B_quant * scale
        b_block_dequant = b_block_fp16 * scales[None, :]

        # Perform dot product
        accumulator = tl.dot(a_block, b_block_dequant, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul_on_the_fly(a, b_quant, scales, activation=""):
    assert a.ndim == b_quant.ndim == 2
    assert a.shape[1] == b_quant.shape[0]
    
    M, K = a.shape
    _, N = b_quant.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    # We let Triton's autotuner supply optimal BLOCK_SIZE_M, BLOCK_SIZE_N, etc.
    matmul_on_the_fly_kernel[grid](
        a, b_quant, c, scales,
        M, N, K,
        a.stride(0), a.stride(1),
        b_quant.stride(0), b_quant.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(1) if scales.ndim == 2 else scales.stride(0),
        ACTIVATION=activation
    )
    return c

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a,b,activation=""):

    assert a.ndim == b.ndim == 2
    assert a.shape[1] == b.shape[0]

    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M,N), device = a.device, dtype=torch.float16)

    # lets get total chunks of c:
    #
    # [a, b, c]
    # [d, e, f]         dimension (M/m_block_size, N/n_block_size), as each letter of the
    # [g, h, i]         C matrix is (block size m, block size n)
    #
    #
    #
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
 
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c


# def test_kernel(size, atol = 1e-2, rtol = 1e-1, device=DEVICE):
#
#     torch.manual_seed(0)
#     assert type(size) == tuple and len(size) == 2
#     a = torch.randn(size, device=DEVICE, dtype=torch.float16)
#     b = torch.randn(size, device=DEVICE, dtype=torch.float16)
#
#     c_tri = matmul(a,b)
#     c_ref = torch.matmul(a,b)
#
#     torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
#
#     print("done")

def run_fp16_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=[128 * i for i in range(2, 33)],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton FP16', 'PyTorch FP16'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='TFLOPS',
            plot_name='matmul-fp16-performance',
            args={},
        )
    )
    def benchmark_fp16(M, N, K, provider):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    print("Running FP16 performance benchmark...")
    benchmark_fp16.run(show_plots=False, print_data=True, save_path='results/plots')
    print("FP16 Benchmark complete!\n")

def run_int8_benchmark():
    @triton.testing.perf_report(
   triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=[128 * i for i in range(2, 33)],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton W8A16', 'PyTorch FP16'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='TFLOPS',
            plot_name='matmul-int8-performance',
            args={},
        )
    )
    def benchmark_int8(M, N, K, provider):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b_fp16 = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        
        # Column-wise symmetric quantization of B to INT8
        b_max = torch.amax(torch.abs(b_fp16), dim=0, keepdim=True)
        scales = torch.clamp(b_max, min=1e-9) / 127.0
        b_quant = torch.clamp(torch.round(b_fp16 / scales), -128, 127).to(torch.int8)
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b_fp16), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_on_the_fly(a, b_quant, scales), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    print("Running INT8 performance benchmark...")
    benchmark_int8.run(show_plots=False, print_data=True, save_path='results/plots')
    print("INT8 Benchmark complete!\n")

if __name__ == "__main__":
    # Correctness test for matmul_on_the_fly (W8A16)
    print("Verifying matmul_on_the_fly (W8A16)...")
    torch.manual_seed(0)
    a_test = torch.randn((128, 256), device=DEVICE, dtype=torch.float16)
    b_test_fp16 = torch.randn((256, 128), device=DEVICE, dtype=torch.float16)
    
    # Column-wise symmetric quantization of B to INT8
    b_max = torch.amax(torch.abs(b_test_fp16), dim=0, keepdim=True)
    scales_test = torch.clamp(b_max, min=1e-9) / 127.0
    b_test_quant = torch.clamp(torch.round(b_test_fp16 / scales_test), -128, 127).to(torch.int8)
    
    # Reference dequantized matmul: A @ (B_quant * scale)
    b_dequant_ref = b_test_quant.to(torch.float16) * scales_test
    c_ref = torch.matmul(a_test, b_dequant_ref)
    
    # Triton matmul_on_the_fly (no activation)
    c_triton = matmul_on_the_fly(a_test, b_test_quant, scales_test)
    
    # Assert correctness (no activation)
    torch.testing.assert_close(c_triton, c_ref, rtol=1e-2, atol=1e-2)
    print("Correctness check for matmul_on_the_fly (no activation) passed successfully!")
    
    # Triton matmul_on_the_fly (with leaky_relu)
    c_ref_act = torch.where(c_ref >= 0, c_ref, c_ref * 0.01)
    c_triton_act = matmul_on_the_fly(a_test, b_test_quant, scales_test, activation="leaky_relu")
    
    # Assert correctness (with leaky_relu)
    torch.testing.assert_close(c_triton_act, c_ref_act, rtol=1e-2, atol=1e-2)
    print("Correctness check for matmul_on_the_fly (with Leaky ReLU) passed successfully!\n")
    
    # Run the benchmarks
    # run_fp16_benchmark()
    run_int8_benchmark()
    

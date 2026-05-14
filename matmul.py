import triton
import triton.language as tl
import torch


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

#
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
#       b = e (shape(BLOCK_SIZE_k, BLOCK_SIZE_N)), memory offset -> B[0:BLOCK_SIZE_K, 0:BLOCK_SIZE_N]
#       acc += dot(a,b)
#
#
# Then it 2:
#
#       a = b (shape(BLOCK_SIZE_M, BLOCK_SIZE_N)), memory offset -> A[0:BLOCK_SIZE_M, K:k+BLOCK_SIZE_K]
#       b = g (shape(BLOCK_SIZE_M, BLOCK_SIZE_N)), memory offset -> B[K:k+BLOCK_SIZE_K, 0:BLOCK_SIZE_N]
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

DEVICE = triton.runtime.driver.active.get_active_torch_device()

import os
os.environ['TRITON_INTERPRETER']="1"






















def matmul(a,b):

    assert a.ndim == b.nidm == 2
    assert a.shape[1] == b.shape[0]

    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M,N), device = a.device, dtype=torch.float16)

    # lets get total chunks of c,
    #
    # [a, b, c]
    # [d, e, f]         dimension (M/m_block_size, N/m_block_size), each letter is (block size m, block size n)
    # [g, h, i]
    #
    
    # this cdiv on the grid means that we are finding the dimensions of the previous target matrix 
    # (which is a and b reduced to blocks)
 
    grid = lambda meta:

        (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']))

    matmul_kernel[grid](

        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

def test_kernel(size, atol = 1e-2, rtol = 1e-1, device=DEVICE):
    
    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn(size, device=DEVICE, dtype=torch.float16)
    b = torch.randn(size, device=DEVICE, dtype=torch.float16)
    
    c_tri = matmul(a,b)
    c_ref = torch.matmul(a,b)

    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)

    print("done")


@triton.jit
def matmul_kernel():

















if __name__ == "__main__":
    #
    # A = torch.rand(3,3)
    # B = torch.rand(3,5)
    #
    # print(naive_matmul(A,B))

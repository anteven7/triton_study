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
# for m in range(0, M, BLOCK_SIZE_M): this is a pararell execution for each block size
#   for n in range(0, N, BLOCK_SIZE_N): this is a pararell execution for each block size 
#     acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#     for k in range(0, K, BLOCK_SIZE_K):
#       a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#       b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#       acc += dot(a, b)
#     C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc


# "Where each iteration of the doubly-nested for-loop is performed by a dedicated triton
# program instance."


if __name__ == "__main__":

    A = torch.rand(3,3)
    B = torch.rand(3,5)

    print(naive_matmul(A,B))

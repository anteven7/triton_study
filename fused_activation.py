import triton
import torch

import triton.language as tl
from triton.runtime import driver


DEVICE = triton.runtime.driver.active.get_active_torch_device()

def naive_sofmax(x):
    
# Let's assume that input is a matrix with dimensions (m,n)
# We compute now a naive_sofmax 
    #
    # This line takes the max of each row of the matrix, for example:
    #
    # tensor([[0.8417, 0.2907],
    #         [0.9507, 0.9943],
    #         [0.6195, 0.2036]], device='cuda:0')
    # tensor([0.8417, 0.9943, 0.6195], device='cuda:0')
    #
    # This produces MN reads (the whole matrix) and M writes (one per row)
    #
    x_max = x.max(dim=1)[0]
    #
    # The following line makes the softmax stable: parameters are usually saved 
    # as fp32, which has a limit of 3.4x10^38. The formula of the sofmax works with
    # exponentials, which means we need to be careful with large number representations.
    # Due to the nature of softmax we can substract C (being C the max(row)) of each row,
    # making number collapse almost impossible. This means MN + N reads(whole matrix x and x_max) and MN writes(matrix z).
    #
    z = x - x_max[:, None]
    #
    # Now we compute the numerator and denominator. MN reads + MN writes and MN reads + M writes respectively. 
    #
    numerator = torch.exp(z) # calculatees the e^x of each element of the input tensor.
    denominator = numerator.sum(dim=1) # pretty straighforward 
    #
    # Las step. MN + M reads and MN writes.
    #
    ret = numerator / denominator[:, None]
    return ret

# If we count how many writes and reads we end up with: 5MN + 2M reads and 3MN + 2M writes, which is hihgly inneficient. 
# We could then build a fused kernel, which reduces this communication with the VRAM diminishing the memory speed bottleneck. 

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def relu_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):

    # First we identify the PIDs, which takes care of the number of data that each program has ownership to compute. 
    # It goes from 0:grid in the axis we define. For example, in a grid (3,), tl.program_id(0) will generate [0, 1, 2].
    # We do this with the rows(axis 0).
    #
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0) # computes the len of programs. if programs are [0,..,N] it returns n+1.
    # 
    

if __name__ == "__main__":

    M = 3
    N = 2

    x = torch.rand(M, N, device=DEVICE)
    print("Original tensor: ", x)
    
    sm = naive_sofmax(x)
    print("Softmaxed tensor: ",sm)










































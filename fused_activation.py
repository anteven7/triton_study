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

def super_naive_leaky_relu(x, alpha=0.01):
    # Clone the tensor so we don't modify the original input unexpectedly
    out = x.clone()
    
    # Flattening makes it easier to loop regardless of shape
    # but let's look at a 2D example for clarity
    for i in range(out.shape[0]):          # Loop over rows
        for j in range(out.shape[1]):      # Loop over columns
            val = out[i, j]
            if val < 0:
                out[i, j] = val * alpha
            # else: out[i, j] = val (stays the same)
    return out

def naive_leaky_relu(x):
    # This is still "naive" but vectorized so it actually finishes
    return torch.where(x > 0, x, x * 0.01)

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
def leaky_relu_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                      num_stages: tl.constexpr, alfa = 0.01):

    # First we identify the PIDs, which take care of the number of data that each program has ownership to compute, depending
    # of the block size. If we choose block_size as the next power of 2 bigger than N, we will compute a row for each PID.
    # It goes from 0:grid in the axis we define. For example, in a grid (3,), tl.program_id(0) will generate [0, 1, 2].
    # We do this with the rows(axis 0).
    #
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0) # computes the len of programs. if programs are [0,..,N] it returns n+1. This is 
    # necessary as we could potentially have more rows than number of programs, so the first program when it finishes 
    # has to step row + row_step to get the next one.
    # 
    # The next for loop is an iteration to get this done:
    #
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages = num_stages): # It means it starts at , goes to maximun 
        # number of rows, iterating by the number of max programs, to treat remaining rows.
        #
        # input_ptr is the memory address of the very first element of the tensor.
        # row_idx is the row we want. 
        # input_row_stride means usually the number of elements in a vector, so when we do row_idx * input_row_stride we 
        # are actually calculating the number of total steps to get to the row we want.
        #
        row_start_ptr = input_ptr + row_idx * input_row_stride
        #
        # now that we are in the chosen row, we do use col_offsets to navigate towards its elements.
        #
        col_offsets = tl.arange(0, BLOCK_SIZE)
        #
        # subsequently, we adjuts the input pointer
        # because broadcasting, we add to every number in vector [0,.., col_offsets] the row_start_ptr, so 
        # we get a tensor of pointers referencing the row we want.
        #
        input_ptrs = row_start_ptr + col_offsets
        #
        # if we have a block size (col_offsets) that is bigger that n cols, the remaining spaces wont compute the next 
        # elements
        #
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask = mask, other = 0.0)
        leaky_row = tl.maximum(row*alfa, row)
        output_row_start_ptr = output_ptr + row_idx * output_row_stride 
        tl.store(output_row_start_ptr + col_offsets, leaky_row, mask=mask)


if __name__ == "__main__":
    
    torch.manual_seed(0)
    x = torch.randn(4096,4096, device=DEVICE)
        
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    n = y.numel()
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Number of software pipelining stages.
    num_stages = 4 
    grid = (n_rows,)
    # y_triton = compute(x)
    # y_torch = torch.nn.functional.leaky_relu(x)
    # 2. Benchmark the kernel call directly
    softmax_triton_ms = triton.testing.do_bench(
        lambda: softmax_kernel[grid](
        y, x, 
        x.stride(0), y.stride(0), 
        n_rows, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE, 
        num_stages=num_stages
    )
)
    softmax_torch_ms = triton.testing.do_bench(
        lambda: naive_sofmax(x)
)
    print(softmax_triton_ms)
    print(softmax_torch_ms)

    relu_kernel_ms = triton.testing.do_bench(
        lambda: leaky_relu_kernel[grid](
        y, x, 
        x.stride(0), y.stride(0), 
        n_rows, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE, 
        num_stages=num_stages
    )
)
    with torch.no_grad():
        
        relu_torch_ms = triton.testing.do_bench(
            lambda: naive_leaky_relu(x)
)

    print(relu_kernel_ms)
    print(relu_torch_ms)






































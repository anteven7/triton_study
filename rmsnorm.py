import os 
os.environ['TRITON_INTERPRETER'] = "1"
import torch
import triton
import triton.language as tl
import math


def naive_rmsnorm(x, w, epsilon=1e-1):
    out = torch.empty_like(x)
    for i in range(len(x)):

        rms = torch.sum(x[i]**2)/len(x[i])
        rms = math.sqrt(rms + epsilon)

        out[i]= (x[i]/rms)*w 

    return out



@triton.jit 
def rmsnorm_kernel_fused(x_pointer, y_pointer,
                          row_stride, epsilon, N, BLOCK_SIZE: tl.constexpr):

    #print("block size", BLOCK_SIZE) # next power of two of cols.
    #print("n of cols",N) # cols

    # by desing we launch a program per row

    row = tl.program_id(0) # the row or program we are in 
    x = x_pointer + row * row_stride
    y = y_pointer + row * row_stride

    # compute mean, this is really interesting

    mean = 0
    # _mean is a vector that we define to save the state as an accumulator. It is block size as we look to loop
    # over the total N block size by block size (if N is big enough). If not i will be size next_power_of_2(N).
    # The interesting thing is that if we have to loop BLOCK_SIZE 4 times, we will add partial sums to the _mean, 
    # being _mean[0] = X[0] + X[BLOCK_SIZE] + X[2BLOCK_SIZE] + X[3BLOCK_SIZE].
    #
    # Also, In Triton, any tensor created inside the kernel (like tl.zeros, tl.arange, etc.) 
    # must have its dimensions defined by a compile-time constant (tl.constexpr).
    _mean = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

    # this loop is kind of triky: We actually defined BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # if we have a next_power_of_2(N) value that can be actually fitted, we use that. Therefore BLOCK SIZE would
    # be bigger and the loop will execute once. If by any chance N is super big, we have to schedule and execute
    # by block size fitting the whole avaliable resources.

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE) # will be an array from 0 to BLOCK_SIZE. If N is massive as we 
        # talked, it would be for example an array of [0, BLOCK_SIZE] it 1, [BLOCK_SIZE, 2BLOCK_SIZE] it 2, ...
        #
        # lets now load the row, summing the actual row pointer (x) to cols, applying the mask.
        #
        a = tl.load(x + cols, mask = cols < N, other = 0.).to(tl.float32)
        #
        # now is when we add to _mean!
        #
        _mean += a * a

    mean = tl.sum(_mean, axis=0)/N

    # now lets compute the sqrt
    rms = tl.sqrt(mean + epsilon)

    

    for off in range(0, N, BLOCK_SIZE):
        # again, getting the chunck of data we are going to process 
        cols = off + tl.arange(0, BLOCK_SIZE) 
        
        # pulling back the values to sram
        a = tl.load(x+cols, mask = cols<N, other = 0.).to(tl.float32)
        
        rmsnorm = a / rms
        
        tl.store(y+cols, rmsnorm, mask=cols<N)


@triton.jit
def rmsnorm_kernel_no_loops(x_pointer, y_pointer,
                          row_stride, epsilon, N, BLOCK_SIZE: tl.constexpr):
    # we make a super big assumption here that is that our max rows would be less than 
    # the normal hidden dimension of 4096, that means we do not need to iterate 
    # the row we are treating as we could include the whole of it into a block size
    row = tl.program_id(0)

    x = x_pointer + row*row_stride
    y = y_pointer + row*row_stride
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols<N

    a = tl.load(x+cols, mask=mask, other=0.).to(tl.float32)
    mean = tl.sum(a*a, axis=0)/N
    rms = tl.sqrt(mean+epsilon)
    output = a / rms
    tl.store(y+cols, output, mask=mask )


def rmsnorm_kernel_test(x, no_loop):

    M,N = x.shape

    grid = (M,)
    
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)
    if no_loop:
        rmsnorm_kernel_no_loops[grid](x,y,x.stride(0), 1e-1, N, BLOCK_SIZE)
    else:
        rmsnorm_kernel_fused[grid](x, y, x.stride(0), 1e-1, N, BLOCK_SIZE) 
   
    return y 


if __name__=='__main__':
    torch.manual_seed(42)
    matrix = torch.randn(10,10, device='cuda')

    print(matrix)

    naive = triton.testing.do_bench(lambda: naive_rmsnorm(matrix,w=1)) 
    triton_loops =  triton.testing.do_bench(lambda: rmsnorm_kernel_test(matrix, False))
    triton_no_loops = triton.testing.do_bench(lambda:rmsnorm_kernel_test(matrix, True))


    print(naive)
    print(triton_loops)
    print(triton_no_loops)

    print(f'triton with loops is {naive/triton_loops} times faster than the naive implementation') 
    print(f'triton with no loops is {naive/triton_no_loops} times faster than the naive implementation') 





























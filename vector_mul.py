import triton
import torch

import triton.language as tl
from triton.runtime import driver


DEVICE = triton.runtime.driver.active.get_active_torch_device()

print(DEVICE)

@triton.jit
def multiply_vectors(x_prt, y_prt, output_prt,
 vector_size, BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE


    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask =  offsets < vector_size

    x = tl.load(x_prt + offsets, mask=mask)
    y = tl.load(y_prt + offsets, mask=mask)

    output = x * y
    
    tl.store(output_prt + offsets, output, mask=mask)
def multiply(x: torch.Tensor, y: torch.Tensor):

    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    n_elements = output.numel()


    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )


    multiply_vectors[grid](x,y,output,n_elements, BLOCK_SIZE=512)

    return output 


if __name__ =="__main__":
    torch.manual_seed(0)
    size = 16

    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device = DEVICE)
    kernel_mul = multiply(x,y)
    naive_mul = x * y 
    print(x)
    print(y)
    print(kernel_mul)
    print(naive_mul)
    print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(naive_mul - kernel_mul))}')

    triton_ms = triton.testing.do_bench(lambda: multiply(x,y)) 
    torch_ms =  triton.testing.do_bench(lambda: x*y)

    print(triton_ms)
    print(torch_ms)


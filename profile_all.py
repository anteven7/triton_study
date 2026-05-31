import torch
import triton
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import Triton implementations
from rmsnorm import rmsnorm_kernel_test, naive_rmsnorm, torch_rmsnorm
from fused_activation import softmax_kernel, naive_sofmax
from matmul import matmul, matmul_on_the_fly

# Quantization is done inline using vectorized PyTorch operations

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def profile_memory_bound_operators():
    """
    Profile RMSNorm and Softmax operators.
    Saves performance comparison charts for Latency and Bandwidth (GB/s).
    """
    print("\n--- Profiling Memory-Bound Operators (Softmax & RMSNorm) ---")
    
    # We sweep column size N (hidden dimension), holding rows M constant at 1024
    M = 1024
    N_sizes = [512, 1024, 2048, 4096, 8192, 12288, 16384]
    
    # Latency arrays
    rmsnorm_naive_lat = []
    rmsnorm_torch_lat = []
    rmsnorm_triton_lat = []
    softmax_torch_lat = []
    softmax_triton_lat = []
    
    for N in N_sizes:
        print(f"Profiling size: M={M}, N={N}...")
        
        # Allocate tensors
        x = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
        w = torch.randn(N, device=DEVICE, dtype=torch.float16)
        y_out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(N)
        
        # --- RMSNorm ---
        naive_rmsnorm_ms = triton.testing.do_bench(lambda: naive_rmsnorm(x, w))
        torch_rmsnorm_ms = triton.testing.do_bench(lambda: torch_rmsnorm(x, w))
        triton_rmsnorm_ms = triton.testing.do_bench(lambda: rmsnorm_kernel_test(x, w, no_loop=True))
        
        rmsnorm_naive_lat.append(naive_rmsnorm_ms)
        rmsnorm_torch_lat.append(torch_rmsnorm_ms)
        rmsnorm_triton_lat.append(triton_rmsnorm_ms)
        
        # --- Softmax ---
        grid = (M,)
        torch_softmax_ms = triton.testing.do_bench(lambda: naive_sofmax(x))
        triton_softmax_ms = triton.testing.do_bench(
            lambda: softmax_kernel[grid](
                y_out, x, x.stride(0), y_out.stride(0), M, N,
                BLOCK_SIZE=BLOCK_SIZE, num_stages=4
            )
        )
        
        softmax_torch_lat.append(torch_softmax_ms)
        softmax_triton_lat.append(triton_softmax_ms)

    # Convert lists to numpy arrays
    N_sizes = np.array(N_sizes)
    rmsnorm_naive_lat = np.array(rmsnorm_naive_lat)
    rmsnorm_torch_lat = np.array(rmsnorm_torch_lat)
    rmsnorm_triton_lat = np.array(rmsnorm_triton_lat)
    softmax_torch_lat = np.array(softmax_torch_lat)
    softmax_triton_lat = np.array(softmax_triton_lat)

    # Calculate VRAM Bandwidth (GB/s)
    # Bandwidth = (Bytes Read + Bytes Written) / (Time in seconds * 1e9)
    # Both operators read FP16 input (M * N * 2 bytes) and write FP16 output (M * N * 2 bytes). Total = 4 * M * N bytes.
    bytes_transferred = 4 * M * N_sizes
    
    rmsnorm_naive_gbps = (bytes_transferred / (rmsnorm_naive_lat * 1e-3)) / 1e9
    rmsnorm_torch_gbps = (bytes_transferred / (rmsnorm_torch_lat * 1e-3)) / 1e9
    rmsnorm_triton_gbps = (bytes_transferred / (rmsnorm_triton_lat * 1e-3)) / 1e9
    softmax_torch_gbps = (bytes_transferred / (softmax_torch_lat * 1e-3)) / 1e9
    softmax_triton_gbps = (bytes_transferred / (softmax_triton_lat * 1e-3)) / 1e9
    
    # Save results to CSV for thesis tables
    with open("results/csvs/memory_bound_results.csv", "w") as f:
        f.write("N,RMSNorm_Naive_ms,RMSNorm_Torch_ms,RMSNorm_Triton_ms,RMSNorm_Naive_GBps,RMSNorm_Torch_GBps,RMSNorm_Triton_GBps,Softmax_Torch_ms,Softmax_Triton_ms,Softmax_Torch_GBps,Softmax_Triton_GBps\n")
        for i in range(len(N_sizes)):
            f.write(f"{N_sizes[i]},{rmsnorm_naive_lat[i]:.4f},{rmsnorm_torch_lat[i]:.4f},{rmsnorm_triton_lat[i]:.4f},"
                    f"{rmsnorm_naive_gbps[i]:.2f},{rmsnorm_torch_gbps[i]:.2f},{rmsnorm_triton_gbps[i]:.2f},"
                    f"{softmax_torch_lat[i]:.4f},{softmax_triton_lat[i]:.4f},{softmax_torch_gbps[i]:.2f},{softmax_triton_gbps[i]:.2f}\n")
    print("Saved memory_bound_results.csv")

    # Plot Latency Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(N_sizes, rmsnorm_naive_lat, 'x--', label='PyTorch Naive (Loops)', color='orange')
    plt.plot(N_sizes, rmsnorm_torch_lat, 'o-', label='PyTorch Vectorized', color='coral')
    plt.plot(N_sizes, rmsnorm_triton_lat, 's-', label='Triton Fused', color='teal')
    plt.title('RMSNorm Latency Comparison (M=1024)')
    plt.xlabel('Hidden Dimension (N)')
    plt.ylabel('Latency (ms)')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(N_sizes, softmax_torch_lat, 'o-', label='PyTorch Naive', color='coral')
    plt.plot(N_sizes, softmax_triton_lat, 's-', label='Triton Fused', color='teal')
    plt.title('Softmax Latency Comparison (M=1024)')
    plt.xlabel('Hidden Dimension (N)')
    plt.ylabel('Latency (ms)')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/memory_bound_latency.png')
    plt.close()
    print("Saved memory_bound_latency.png")

    # Plot Bandwidth Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(N_sizes, rmsnorm_naive_gbps, 'x--', label='PyTorch Naive (Loops)', color='orange')
    plt.plot(N_sizes, rmsnorm_torch_gbps, 'o-', label='PyTorch Vectorized', color='coral')
    plt.plot(N_sizes, rmsnorm_triton_gbps, 's-', label='Triton Fused', color='teal')
    plt.title('RMSNorm Memory Bandwidth Utilization')
    plt.xlabel('Hidden Dimension (N)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(N_sizes, softmax_torch_gbps, 'o-', label='PyTorch Naive', color='coral')
    plt.plot(N_sizes, softmax_triton_gbps, 's-', label='Triton Fused', color='teal')
    plt.title('Softmax Memory Bandwidth Utilization')
    plt.xlabel('Hidden Dimension (N)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/memory_bound_bandwidth.png')
    plt.close()
    print("Saved memory_bound_bandwidth.png")


def profile_compute_bound_operators():
    """
    Profile Matrix Multiplication operators (FP16 vs. W8A16).
    Saves performance comparison charts for Throughput (TFLOPS).
    """
    print("\n--- Profiling Compute-Bound Operators (Matmul FP16 vs. W8A16) ---")
    
    # Square matrix sweeps
    sizes = [256, 512, 1024, 1536, 2048, 2560, 3072, 4096]
    
    torch_fp16_tflops = []
    triton_fp16_tflops = []
    triton_w8a16_tflops = []
    
    for size in sizes:
        print(f"Profiling Matrix size: {size}x{size}x{size}...")
        M = N = K = size
        
        # 1. Allocate tensors
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b_fp16 = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        
        # Column-wise symmetric quantization of B to INT8
        b_max = torch.amax(torch.abs(b_fp16), dim=0, keepdim=True)
        scales = (torch.clamp(b_max, min=1e-9) / 127.0).to(torch.float16)
        b_quant = torch.clamp(torch.round(b_fp16 / scales), -128, 127).to(torch.int8)
        
        # Precompute reference dequantized tensor to isolate GEMM compute speedup
        b_dequant_ref = b_quant.to(torch.float16) * scales

        # 2. Run Benchmarks
        torch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b_fp16))
        triton_fp16_ms = triton.testing.do_bench(lambda: matmul(a, b_fp16))
        triton_w8a16_ms = triton.testing.do_bench(lambda: matmul_on_the_fly(a, b_quant, scales))
        
        # Calculate TFLOPS: 2 * M * N * K FLOPs
        flops = 2 * M * N * K
        
        torch_fp16_tflops.append((flops / (torch_ms * 1e-3)) / 1e12)
        triton_fp16_tflops.append((flops / (triton_fp16_ms * 1e-3)) / 1e12)
        triton_w8a16_tflops.append((flops / (triton_w8a16_ms * 1e-3)) / 1e12)

    # Save to CSV for thesis tables
    with open("results/csvs/matmul_results.csv", "w") as f:
        f.write("Size,Torch_FP16_TFLOPS,Triton_FP16_TFLOPS,Triton_W8A16_TFLOPS\n")
        for i in range(len(sizes)):
            f.write(f"{sizes[i]},{torch_fp16_tflops[i]:.4f},{triton_fp16_tflops[i]:.4f},{triton_w8a16_tflops[i]:.4f}\n")
    print("Saved matmul_results.csv")

    # Plot Matmul Performance in TFLOPS
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, torch_fp16_tflops, 'o-', label='PyTorch FP16', color='coral')
    plt.plot(sizes, triton_fp16_tflops, 's-', label='Triton FP16', color='teal')
    plt.plot(sizes, triton_w8a16_tflops, '^-', label='Triton W8A16 (Dequant on-the-fly)', color='indigo')
    plt.title('GEMM Performance Comparison')
    plt.xlabel('Matrix Size (M=N=K)')
    plt.ylabel('Throughput (TFLOPS)')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/matmul_performance.png')
    plt.close()
    print("Saved matmul_performance.png")


if __name__ == "__main__":
    profile_memory_bound_operators()
    profile_compute_bound_operators()
    print("\nAll profiling runs completed successfully! Verification files generated in local folder.")

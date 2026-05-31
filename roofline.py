import numpy as np
import matplotlib.pyplot as plt
import os

# --- Hardware Constants (NVIDIA GeForce GTX 1650 Mobile / Max-Q Representative Specs) ---
PEAK_COMPUTE_TFLOPS = 5.6  # Peak FP16 performance
PEAK_BANDWIDTH_GBPS = 128.0 # Peak VRAM memory bandwidth (GB/s)

def calculate_theoretical_roofline(x_range):
    """
    Computes the attainable performance in TFLOPS for a given arithmetic intensity range.
    """
    # Performance = min(Peak Compute, Intensity * Bandwidth)
    # Bandwidth in TBytes/s = PEAK_BANDWIDTH_GBPS / 1000.0
    bw_tbps = PEAK_BANDWIDTH_GBPS / 1000.0
    attainable = np.minimum(PEAK_COMPUTE_TFLOPS, x_range * bw_tbps)
    return attainable

def load_and_plot_kernels():
    """
    Reads benchmarking CSV results and plots empirical performance points on the roofline graph.
    """
    # Create x-axis range for intensity (log scale)
    x_intensity = np.logspace(-2, 4, 1000)
    y_attainable = calculate_theoretical_roofline(x_intensity)
    
    plt.figure(figsize=(10, 6))
    
    # 1. Plot the Roofline boundary
    plt.loglog(x_intensity, y_attainable, 'k-', linewidth=2.5, label='Theoretical Roofline')
    plt.axhline(y=PEAK_COMPUTE_TFLOPS, color='red', linestyle='--', alpha=0.7, label=f'Compute Peak ({PEAK_COMPUTE_TFLOPS} TFLOPS)')
    
    # Shade the unattainable region
    plt.fill_between(x_intensity, y_attainable, 10, color='gray', alpha=0.1, label='Unattainable Region')

    # Load Memory-Bound Results (Softmax / RMSNorm)
    if os.path.exists("results/csvs/memory_bound_results.csv"):
        print("Found results/csvs/memory_bound_results.csv, loading empirical points...")
        data = np.genfromtxt("results/csvs/memory_bound_results.csv", delimiter=",", names=True)
        # We plot the largest size measured (e.g. N = 16384) to represent operational limits
        last_idx = -1
        N = data['N'][last_idx]
        M = 1024
        
        # --- RMSNorm Point ---
        # AI = 4 * M * N / (4 * M * N + 2 * N) approx 1.0 FLOP/byte
        ai_rmsnorm = (4 * M * N) / (4 * M * N + 2 * N)
        # Throughput = FLOPs / (Latency * 1e-3) in TFLOPS
        flops_rmsnorm = 4 * M * N
        tflops_rmsnorm_naive = (flops_rmsnorm / (data['RMSNorm_Naive_ms'][last_idx] * 1e-3)) / 1e12
        tflops_rmsnorm_torch = (flops_rmsnorm / (data['RMSNorm_Torch_ms'][last_idx] * 1e-3)) / 1e12
        tflops_rmsnorm_triton = (flops_rmsnorm / (data['RMSNorm_Triton_ms'][last_idx] * 1e-3)) / 1e12
        
        plt.scatter(ai_rmsnorm, tflops_rmsnorm_naive, color='orange', marker='x', s=100, zorder=5, label='RMSNorm PyTorch Naive')
        plt.scatter(ai_rmsnorm, tflops_rmsnorm_torch, color='coral', marker='o', s=100, zorder=5, label='RMSNorm PyTorch Vectorized')
        plt.scatter(ai_rmsnorm, tflops_rmsnorm_triton, color='teal', marker='s', s=100, zorder=5, label='RMSNorm Triton (Fused)')
        plt.annotate('RMSNorm', (ai_rmsnorm, tflops_rmsnorm_triton), textcoords="offset points", xytext=(-15, 10), ha='center', weight='bold')

        # --- Softmax Point ---
        # AI = 5 * M * N / (4 * M * N) = 1.25 FLOP/byte
        ai_softmax = 1.25
        flops_softmax = 5 * M * N
        tflops_softmax_torch = (flops_softmax / (data['Softmax_Torch_ms'][last_idx] * 1e-3)) / 1e12
        tflops_softmax_triton = (flops_softmax / (data['Softmax_Triton_ms'][last_idx] * 1e-3)) / 1e12
        
        plt.scatter(ai_softmax, tflops_softmax_torch, color='coral', marker='o', s=100, zorder=5)
        plt.scatter(ai_softmax, tflops_softmax_triton, color='teal', marker='s', s=100, zorder=5)
        plt.annotate('Softmax', (ai_softmax, tflops_softmax_triton), textcoords="offset points", xytext=(15, 10), ha='center', weight='bold')
    else:
        print("results/csvs/memory_bound_results.csv not found. Please run profile_all.py first to get empirical points.")
        # Plot default representative points if csv is missing
        plt.scatter(1.0, 0.05, color='coral', marker='o', s=80, alpha=0.5, label='RMSNorm PyTorch (Est.)')
        plt.scatter(1.0, 0.35, color='teal', marker='s', s=80, alpha=0.5, label='RMSNorm Triton (Est.)')
        plt.scatter(1.25, 0.06, color='coral', marker='o', s=80, alpha=0.5)
        plt.scatter(1.25, 0.42, color='teal', marker='s', s=80, alpha=0.5)

    # Load Compute-Bound Results (Matmul FP16 / W8A16)
    if os.path.exists("results/csvs/matmul_results.csv"):
        print("Found results/csvs/matmul_results.csv, loading empirical points...")
        data = np.genfromtxt("results/csvs/matmul_results.csv", delimiter=",", names=True)
        # Plot the largest matrix size measured (e.g. Size = 4096)
        last_idx = -1
        size = data['Size'][last_idx]
        M = N = K = size
        
        # --- Matmul FP16 ---
        # AI = M * N * K / (M * K + K * N + M * N) FLOP/byte = Size / 3
        ai_fp16 = size / 3.0
        tflops_matmul_torch = data['Torch_FP16_TFLOPS'][last_idx]
        tflops_matmul_triton = data['Triton_FP16_TFLOPS'][last_idx]
        
        plt.scatter(ai_fp16, tflops_matmul_torch, color='coral', marker='o', s=100, zorder=5, label='Matmul FP16 PyTorch')
        plt.scatter(ai_fp16, tflops_matmul_triton, color='teal', marker='s', s=100, zorder=5, label='Matmul FP16 Triton')
        plt.annotate(f'Matmul FP16 ({int(size)})', (ai_fp16, tflops_matmul_triton), textcoords="offset points", xytext=(0, -15), ha='center', weight='bold')

        # --- Matmul W8A16 ---
        # AI = 2 * M * N * K / (2 * M * K + K * N + 2 * M * N) = 2 * Size / 4.5 = Size / 2.25
        ai_w8a16 = size / 2.25
        tflops_matmul_w8a16 = data['Triton_W8A16_TFLOPS'][last_idx]
        
        plt.scatter(ai_w8a16, tflops_matmul_w8a16, color='indigo', marker='^', s=120, zorder=5, label='Matmul W8A16 Triton')
        plt.annotate(f'Matmul W8A16 ({int(size)})', (ai_w8a16, tflops_matmul_w8a16), textcoords="offset points", xytext=(20, 10), ha='center', weight='bold')
    else:
        print("results/csvs/matmul_results.csv not found. Please run profile_all.py first to get empirical points.")
        # Plot default representative points if csv is missing
        plt.scatter(1365.3, 2.5, color='coral', marker='o', s=80, alpha=0.5, label='Matmul FP16 PyTorch (Est.)')
        plt.scatter(1365.3, 3.8, color='teal', marker='s', s=80, alpha=0.5, label='Matmul FP16 Triton (Est.)')
        plt.scatter(1820.4, 4.2, color='indigo', marker='^', s=100, alpha=0.5, label='Matmul W8A16 Triton (Est.)')

    # Formatting plot
    plt.title('Empirical Roofline Model (GTX 1650 Mobile)', fontsize=14, weight='bold')
    plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPS)', fontsize=12)
    
    # Limits and labels
    plt.xlim(1e-1, 1e4)
    plt.ylim(1e-3, 10)
    plt.grid(True, which="both", ls="--", color='gray', alpha=0.5)
    plt.legend(loc='lower right', framealpha=0.9)
    
    # Add info box
    info_text = (
        f"GPU Limits:\n"
        f"Memory Bandwidth: {PEAK_BANDWIDTH_GBPS} GB/s\n"
        f"FP16 Compute Peak: {PEAK_COMPUTE_TFLOPS} TFLOPS\n"
        f"Ridge Point: {PEAK_COMPUTE_TFLOPS / (PEAK_BANDWIDTH_GBPS/1000.0):.2f} FLOP/Byte"
    )
    plt.gca().text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('results/plots/roofline_model.png', dpi=300)
    plt.close()
    print("Saved results/plots/roofline_model.png successfully!")

if __name__ == "__main__":
    load_and_plot_kernels()

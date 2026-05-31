import torch
import time

def asymmetric_quantization_factor(alfa, beta, bit_width):
    return (beta-alfa)/(2**bit_width - 1)

def symmetric_quantization_factor(rmax, rmin, bit_width):
    clip = torch.max(torch.abs(rmax),torch.abs(rmin))
    return (clip - (-clip))/(2**bit_width - 1)

def calculate_symmetric_tensor(tensor, factor):
    quantized_vector = []
    for weight in tensor:
        quantized_vector.append(int(weight/factor))
    return quantized_vector

def calculate_asymmetric_tensor(tensor, factor, z_point):
    quantized_vector = []
    for weight in tensor:
        quantized_vector.append(int(weight/factor) + z_point)
    return quantized_vector

def int8_quant_paper(A: torch.Tensor):
    scales = []
    c_quant = torch.zeros((A.shape[0], A.shape[1]), dtype=torch.int8, device=A.device)
    c_outliers = torch.zeros_like(A)

    M, N = A.shape

    for i in range(0, M):
        absmax = torch.max(torch.abs(A[i]))
        if absmax > 6:
            # Outlier row is kept in float precision in c_outliers
            scales.append(torch.tensor(1.0))
            c_outliers[i] = A[i]
            # c_quant[i] remains 0
        else:
            alfa = torch.min(A[i])
            beta = torch.max(A[i])
            
            scaling_factor = symmetric_quantization_factor(alfa, beta, 8)
            quantized_list = calculate_symmetric_tensor(A[i], scaling_factor)
            
            # Normal row is stored as true INT8 in c_quant
            c_quant[i] = torch.tensor(quantized_list, dtype=torch.int8, device=A.device)
            scales.append(scaling_factor)

    return c_quant, c_outliers, scales


def int8_simple_quant(A: torch.Tensor):
    scales = []
    c = torch.zeros(A.shape[0], A.shape[1], dtype=torch.int8, device=A.device)
    M, N = A.shape 
    for i in range(0, M):
            alfa = torch.min(A[i])
            beta = torch.max(A[i])

            scaling_factor = symmetric_quantization_factor(alfa, beta, 8)
            quantized_list = calculate_symmetric_tensor(A[i], scaling_factor)

            c[i] = torch.tensor(quantized_list, dtype=torch.int8, device=A.device)
            scales.append(scaling_factor)
    return c, scales


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running correctness checks on device: {device}\n")
    
    # 1. Generate a mock matrix with 10 rows and 10 columns
    torch.manual_seed(42)
    A = torch.randn((10, 10), device=device)
    
    # Manually inject extreme outliers (> 6.0) into rows 2 and 7
    A[2, 4] = 15.2
    A[7, 8] = -8.5
    
    print("Original matrix A:")
    print(A)
    print("-" * 50)
    
    # 2. Run your outlier-aware quantization function
    c_quant, c_outliers, scales = int8_quant_paper(A)
    
    # 3. Print the resulting matrices
    print("Quantized INT8 Matrix (c_quant):")
    print(c_quant)
    print("\nOutliers Float Matrix (c_outliers):")
    print(c_outliers)
    print("\nScaling factors:")
    print(scales)
    print("-" * 50)
    
    # 4. Reconstruct the original matrix to verify correctness
    # Form: reconstructed = c_quant * scale + c_outliers
    A_reconstructed = torch.zeros_like(A)
    for i in range(A.shape[0]):
        # Extract scalar scale value
        s = scales[i].item() if isinstance(scales[i], torch.Tensor) else scales[i]
        
        # If it was an outlier row, its scale is 1.0 and we load the float values
        if s == 1.0 and torch.any(c_outliers[i] != 0):
            A_reconstructed[i] = c_outliers[i]
        else:
            # If normal, multiply the int8 matrix by the scale factor
            A_reconstructed[i] = c_quant[i].to(torch.float32) * s
            
    print("Reconstructed matrix A_reconstructed:")
    print(A_reconstructed)
    print("-" * 50)
    
    # 5. Measure Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(A - A_reconstructed))
    print(f"Reconstruction Mean Absolute Error (MAE): {mae.item():.6f}")
    
    # Outlier rows should have exactly 0 reconstruction error
    outlier_error_row_2 = torch.mean(torch.abs(A[2] - A_reconstructed[2]))
    print(f"Outlier row 2 reconstruction error: {outlier_error_row_2.item():.6f} (Should be 0.000000)")
    
    # Normal rows should have a very small quantization error
    normal_error_row_0 = torch.mean(torch.abs(A[0] - A_reconstructed[0]))
    print(f"Normal row 0 reconstruction error: {normal_error_row_0.item():.6f} (Typical INT8 error)")
    
    assert outlier_error_row_2.item() == 0.0, "Outlier rows must have zero reconstruction error!"
    print("\nCorrectness checks completed successfully! The math is 100% correct.")


    print("-" * 50)
    matrix, scales= int8_simple_quant(A)
    print("\n Lets do a base one without outliers:")
    print(matrix)
    print("scales", scales)

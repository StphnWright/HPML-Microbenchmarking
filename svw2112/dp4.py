import argparse
import time
import numpy as np

def dp(N, A, B):
    R = 0.0
    for j in range(0, N):
        R += A[j] * B[j]
    return R

def measure_dot_product(N, A, B):
    start_time = time.monotonic()
    result = dp(N, A, B)
    elapsed_time = time.monotonic() - start_time
    
    return result, elapsed_time

def compute_metrics(N, elapsed_time):
    bandwidth = (N * 2 * 4 / 1_073_741_824) / elapsed_time
    flops = N * 2 / elapsed_time
    return bandwidth, flops

def main():
    parser = argparse.ArgumentParser(description='Microbenchmark: Compute dot product of two numpy arrays.')
    parser.add_argument('vecsize', type=int, help='Vector size (N)')
    parser.add_argument('measurements', type=int, help='Number of measurements')
    args = parser.parse_args()

    # Print the vector size and number of measurements
    print(f"Vector size: {args.vecsize} with {args.measurements} measurements.")

    A = np.ones(args.vecsize, dtype=np.float32)
    B = np.ones(args.vecsize, dtype=np.float32)
    
    accumulated_time = 0
    
    for i in range(args.measurements):
        result, elapsed_time = measure_dot_product(args.vecsize, A, B)
        
        if i >= args.measurements // 2:
            accumulated_time += elapsed_time
            
        bandwidth, flops = compute_metrics(args.vecsize, elapsed_time)
        print(f'R: {int(result)} <T>: {elapsed_time:.6f} sec B: {bandwidth:.3f} GB/sec F: {flops:.3f} FLOP/sec')

    mean_time = accumulated_time / (args.measurements // 2)
    avg_bandwidth, avg_flops = compute_metrics(args.vecsize, mean_time)
    print(f"N: {args.vecsize} <T>: {mean_time:.6f} sec B: {avg_bandwidth:.3f} GB/sec F: {avg_flops:.3f} FLOP/sec")

if __name__ == '__main__':
    main()
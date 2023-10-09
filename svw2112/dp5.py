import argparse
import time
import numpy as np

def compute_dot_product(A, B):
    return np.dot(A, B)

def time_and_metrics(N, compute_func):
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start_time = time.monotonic()
    result = compute_func(A, B)
    end_time = time.monotonic()

    elapsed_time = end_time - start_time
    bandwidth = (N * 2 * 4 / 1_073_741_824) / elapsed_time
    flops = N * 2 / elapsed_time

    return result, elapsed_time, bandwidth, flops

def measure_performance(N, measurements):
    total_time, total_bandwidth, total_flops = 0, 0, 0

    for _ in range(measurements):
        result, elapsed_time, bandwidth, flops = time_and_metrics(N, compute_dot_product)
        
        if _ >= measurements // 2:
            total_time += elapsed_time
            total_bandwidth += bandwidth
            total_flops += flops

        print(f'R: {int(result)} <T>: {elapsed_time:.6f} sec B: {bandwidth:.3f} GB/sec F: {flops:.3f} FLOP/sec')

    avg_time = total_time / (measurements // 2)
    avg_bandwidth = total_bandwidth / (measurements // 2)
    avg_flops = total_flops / (measurements // 2)

    print(f"N: {N} <T>: {avg_time:.6f} sec B: {avg_bandwidth:.3f} GB/sec F: {avg_flops:.3f} FLOP/sec")

def main():
    parser = argparse.ArgumentParser(description='Measure dot product performance.')
    parser.add_argument('vecsize', type=int, help='Size of vectors.')
    parser.add_argument('measurements', type=int, help='Number of measurements to perform.')
    args = parser.parse_args()

    # Print the vector size and number of measurements
    print(f"Vector size: {args.vecsize} with {args.measurements} measurements.")

    measure_performance(args.vecsize, args.measurements)

if __name__ == '__main__':
    main()
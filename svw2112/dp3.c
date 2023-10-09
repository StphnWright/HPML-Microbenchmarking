#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl_cblas.h>

#define GB (1073741824.0) // Constant for gigabyte conversion

// Compute the dot product of two vectors using the Intel MKL library
float bdp(long N, float *pA, float *pB)
{
    float R = cblas_sdot(N, pA, 1, pB, 1);
    return R;
}

// Initialize two vectors with all elements set to 1
void initialize_vectors(long N, float *vecA, float *vecB)
{
    for (long idx = 0; idx < N; idx++)
    {
        vecA[idx] = 1.0;
        vecB[idx] = 1.0;
    }
}

int main(int argc, char *argv[])
{
    // Check for the correct number of command-line arguments
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <vector_size> <measurements>\n", argv[0]);
        return 1;
    }

    // Convert command-line arguments to respective data types
    long vector_size = atol(argv[1]);
    int num_measurements = atoi(argv[2]);

    // Print the vector size and number of measurements
    printf("Vector size: %ld with %d measurements.\n", vector_size, num_measurements);

    // Allocate memory for the two vectors
    float *vecA = (float *)malloc(vector_size * sizeof(float));
    float *vecB = (float *)malloc(vector_size * sizeof(float));

    // Initialize vectors to default values
    initialize_vectors(vector_size, vecA, vecB);

    struct timespec start, end;
    double cumulative_time = 0.0;

    // Perform the dot product multiple times and measure its performance
    for (int measure = 0; measure < num_measurements; measure++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start); // Start the timer
        float result = bdp(vector_size, vecA, vecB);
        clock_gettime(CLOCK_MONOTONIC, &end); // Stop the timer

        // Calculate elapsed time
        double elapsed_time = ((double)end.tv_sec + 1.0e-9 * end.tv_nsec) -
                              ((double)start.tv_sec + 1.0e-9 * start.tv_nsec);

        // Accumulate time from the second half measurements for averages
        if (measure >= num_measurements / 2)
        {
            cumulative_time += elapsed_time;
        }

        // Compute bandwidth and flops
        double bandwidth = (vector_size * 2 * sizeof(float) / GB) / elapsed_time;
        double flops = vector_size * 2 / elapsed_time;

        printf("R: %ld <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n",
               (long)result, elapsed_time, bandwidth, flops);
    }

    // Calculate and print the averages for the second half measurements
    double mean_time = cumulative_time / (num_measurements / 2);
    double mean_bandwidth = (vector_size * 2 * sizeof(float) / GB) / mean_time;
    double mean_flops = vector_size * 2 / mean_time;

    printf("N: %li <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n",
           vector_size, mean_time, mean_bandwidth, mean_flops);

    // Free the allocated memory
    free(vecA);
    free(vecB);

    return 0;
}
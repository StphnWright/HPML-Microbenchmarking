#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GB (1073741824.0)   // Constant for gigabyte conversion
#define CHUNK_SIZE 16777216 // Chunk size for precision control

// Compute the dot product of two vectors using loop unrolling for optimization
float dpunroll(long N, float *pA, float *pB)
{
    float R = 0.0;
    int j;
    for (j = 0; j < N; j += 4)
        R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] +
             pA[j + 2] * pB[j + 2] + pA[j + 3] * pB[j + 3];
    return R;
}

// Compute the dot product of two vectors in chunks to avoid precision issues
float dpunroll_chunked(long N, float *pA, float *pB)
{
    float R = 0.0;
    for (long i = 0; i < N; i += CHUNK_SIZE)
    {
        long current_chunk_size = (N - i > CHUNK_SIZE) ? CHUNK_SIZE : (N - i);
        R += dpunroll(current_chunk_size, pA + i, pB + i);
    }
    return R;
}

// Initialize two vectors with all elements set to 1
void initialize_vectors(long N, float *vecA, float *vecB)
{
    for (long i = 0; i < N; i++)
    {
        vecA[i] = 1.0;
        vecB[i] = 1.0;
    }
}

int main(int argc, char *argv[])
{
    // Check for the correct number of command-line arguments
    if (argc < 3)
    {
        printf("Usage: %s [vector size] [number of measurements]\n", argv[0]);
        return 1;
    }

    // Convert command-line arguments to respective data types
    long vec_size = atol(argv[1]);
    int num_measurements = atoi(argv[2]);
    printf("Vector size: %ld with %d measurements.\n", vec_size, num_measurements);

    // Allocate memory for the two vectors
    float *vecA = (float *)malloc(vec_size * sizeof(float));
    float *vecB = (float *)malloc(vec_size * sizeof(float));

    // Initialize vectors to default values
    initialize_vectors(vec_size, vecA, vecB);

    struct timespec start_time, end_time;
    double cumulative_time = 0;
    float dp_result;

    // Perform the dot product multiple times and measure its performance
    for (int i = 0; i < num_measurements; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start_time); // Start the timer

        // Chunked version for large vectors, unrolled for smaller vectors
        if (vec_size > CHUNK_SIZE)
            dp_result = dpunroll_chunked(vec_size, vecA, vecB);
        else
            dp_result = dpunroll(vec_size, vecA, vecB);

        clock_gettime(CLOCK_MONOTONIC, &end_time); // Stop the timer

        // Calculate elapsed time
        double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                              (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

        // Accumulate time from the second half measurements for averages
        if (i >= num_measurements / 2)
            cumulative_time += elapsed_time;

        // Compute bandwidth and flops
        double bandwidth = (vec_size * 2 * sizeof(float) / GB) / elapsed_time;
        double flops = vec_size * 2 / elapsed_time;

        printf("R: %ld <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n",
               (long)dp_result, elapsed_time, bandwidth, flops);
    }

    // Calculate and print the averages for the second half measurements
    double average_time = cumulative_time / (num_measurements / 2);
    double average_bandwidth = (vec_size * 2 * sizeof(float) / GB) / average_time;
    double average_flops = vec_size * 2 / average_time;

    printf("N: %li <T>: %.6f sec B: %.3f GB/sec F: %.3f FLOP/sec\n",
           vec_size, average_time, average_bandwidth, average_flops);

    // Free the allocated memory
    free(vecA);
    free(vecB);

    return 0;
}
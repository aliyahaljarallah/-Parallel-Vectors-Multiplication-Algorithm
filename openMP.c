#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000000    // Size of the vectors
#define NUM_TRIALS 10 // Number of trials to average

// Function to generate a random float between 0 and 1
float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);  // Disable stdout buffering

    // Print vector size and number of processors
    printf("Vector size: %d\n", N);
    int num_procs = omp_get_num_procs();  // Get the number of processors available
    printf("Number of processors: %d\n", num_procs);

    // Sequentially create two random vectors
    float *vector1 = (float *)malloc(N * sizeof(float));
    float *vector2 = (float *)malloc(N * sizeof(float));

    srand(42);  // Seed the random number generator for reproducibility
    for (int i = 0; i < N; i++) {
        vector1[i] = rand_float();
        vector2[i] = rand_float();
    }

    // Print the first 5 values of each vector for debugging purposes
    printf("First 5 values of vector1: ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", vector1[i]);
    }
    printf("\n");

    printf("First 5 values of vector2: ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", vector2[i]);
    }
    printf("\n");

    // Sequential Euclidean distance calculation for verification
    double seq_distance = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = (double)vector1[i] - (double)vector2[i];
        seq_distance += diff * diff;
    }
    seq_distance = sqrt(seq_distance);
    printf("Sequential Euclidean Distance (for verification): %.8f\n", seq_distance);

    // Run trials for each thread count independently
    for (int threads = 2; threads <= 16; threads *= 2) {  // Now includes 2, 4, 8, and 16 threads
        omp_set_num_threads(threads);  // Set number of threads for OpenMP
        printf("Running with %d threads:\n", threads);

        double total_time = 0.0;

        // Run multiple trials to calculate the Euclidean distance
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            double start_time = omp_get_wtime();  // Start timing before the computation

            // Parallel computation of Euclidean distance using OpenMP
            double euclidean_distance = 0.0;  // Using double for precision
            #pragma omp parallel for reduction(+:euclidean_distance)
            for (int i = 0; i < N; i++) {
                double diff = (double)vector1[i] - (double)vector2[i];
                euclidean_distance += diff * diff;
            }

            euclidean_distance = sqrt(euclidean_distance);  // Take square root of the sum of squares

            double end_time = omp_get_wtime();  // End timing after computation
            double time_elapsed = end_time - start_time;  // Time taken for this trial
            total_time += time_elapsed;  // Accumulate time for averaging

            // Print results for each trial
            printf("Trial %d - Euclidean Distance: %.8f, Computation Time: %.6f seconds\n", trial + 1, euclidean_distance, time_elapsed);
        }

        // Print the average computation time for this thread count
        printf("Average Computation Time with %d threads: %.6f seconds\n\n", threads, total_time / NUM_TRIALS);
    }

    // Free dynamically allocated memory
    free(vector1);
    free(vector2);

    return 0;
}

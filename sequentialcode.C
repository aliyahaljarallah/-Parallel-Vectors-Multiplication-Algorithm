#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Function to calculate the Euclidean distance between two vectors
double euclidean_distance(double *a, double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff; // Sum of squares of differences
    }
    return sqrt(sum); // Return the square root of the sum
}

int main() {
    int n;
    printf("Enter vector length: ");
    scanf("%d", &n); // Get vector length from user input
    if (n <= 0) {
        printf("Vector length must be a positive integer.\n");
        return 1;
    }

    // Seed random number generator
    srand(time(NULL));

    // Allocate memory for two vectors
    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    if (a == NULL || b == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    double total_time = 0.0; // Variable to hold total time
    int runs = 10; // Number of runs

    for (int r = 0; r < runs; r++) {
        // Generate random values for vectors
        for (int i = 0; i < n; i++) {
            a[i] = (double)(rand() % 100); // Random values between 0 and 99
            b[i] = (double)(rand() % 100);
        }

        // Start timing
        clock_t start_time = clock();

        // Calculate the Euclidean distance
        double distance = euclidean_distance(a, b, n);

        // End timing
        clock_t end_time = clock();

        // Calculate elapsed time in seconds
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        total_time += elapsed_time; // Accumulate total time

        // Print the result for this run
        printf("Run %d: Euclidean distance = %f, Time taken = %f seconds\n", r + 1, distance, elapsed_time);
    }

    // Calculate and print average time
    double average_time = total_time / runs;
    printf("Average time taken over %d runs: %f seconds\n", runs, average_time);

    // Free allocated memory
    free(a);
    free(b);

    return 0;
}

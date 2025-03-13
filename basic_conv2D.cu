#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

#define R 1
#define WIDTH 5
#define HEIGHT 5

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void conv_2d(float *N, float *F, float *P) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < WIDTH && row < HEIGHT) {
        float pValue = 0.0f;

        // Loop over the filter
        for (int r = 0; r < (2 * R + 1); r++) {
            for (int c = 0; c < (2 * R + 1); c++) {
                int inCol = col - R + c;
                int inRow = row - R + r;
                
                // Boundary check
                if (inCol >= 0 && inCol < WIDTH && inRow >= 0 && inRow < HEIGHT) {
                    pValue += N[inRow * WIDTH + inCol] * F[r * (2 * R + 1) + c];
                }
            }
        }
        P[row * WIDTH + col] = pValue;
    }
}

int main() {
    float *N, *F, *P;
    int filter_width = (2 * R + 1);

    // Allocate host memory
    N = (float*)malloc(HEIGHT * WIDTH * sizeof(float));
    P = (float*)malloc(HEIGHT * WIDTH * sizeof(float));
    F = (float*)malloc(filter_width * filter_width * sizeof(float));

    // Initialize input matrix and filter
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            N[i * WIDTH + j] = rand() % 5;
        }
    }
    for (int i = 0; i < filter_width; i++) {
        for (int j = 0; j < filter_width; j++) {
            F[i * filter_width + j] = rand() % 3;
        }
    }

    float *N_d, *F_d, *P_d;

    // Allocate device memory
    cudaCheckError(cudaMalloc((void**)&N_d, HEIGHT * WIDTH * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&P_d, HEIGHT * WIDTH * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&F_d, filter_width * filter_width * sizeof(float)));

    // Copy data to device
    cudaCheckError(cudaMemcpy(N_d, N, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(F_d, F, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(WIDTH / 16.0), ceil(HEIGHT / 16.0), 1);

    // Launch the kernel
    conv_2d<<<dimGrid, dimBlock>>>(N_d, F_d, P_d);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(P, P_d, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Print input matrix (N)
    cout << "N:\n";
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            cout << N[i * WIDTH + j] << " ";
        }
        cout << "\n";
    }

    // Print filter matrix (F)
    cout << "\nF:\n";
    for (int i = 0; i < filter_width; i++) {
        for (int j = 0; j < filter_width; j++) {
            cout << F[i * filter_width + j] << " ";
        }
        cout << "\n";
    }

    // Print output matrix (P)
    cout << "\nP:\n";
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            cout << P[i * WIDTH + j] << " ";
        }
        cout << "\n";
    }

    // Free device memory
    cudaFree(N_d);
    cudaFree(P_d);
    cudaFree(F_d);

    // Free host memory
    free(N);
    free(F);
    free(P);

    return 0;
}

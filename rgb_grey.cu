#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>  // OpenCV for image handling

using namespace std;
using namespace cv;

__global__ void rgbToGrayKernel(uchar3* rgbImage, unsigned char* grayImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;  // 1D index from 2D coordinates
        uchar3 pixel = rgbImage[idx];

        // Convert RGB to Grayscale
        grayImage[idx] = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    }
}

void convertRGBToGrayCUDA(Mat& inputImage, Mat& outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    uchar3* d_rgbImage;
    unsigned char* d_grayImage;

    size_t numPixels = width * height;
    size_t rgbSize = numPixels * sizeof(uchar3);
    size_t graySize = numPixels * sizeof(unsigned char);

    // Allocate memory on GPU
    cudaMalloc((void**)&d_rgbImage, rgbSize);
    cudaMalloc((void**)&d_grayImage, graySize);

    // Copy input image to GPU
    cudaMemcpy(d_rgbImage, inputImage.ptr<uchar3>(), rgbSize, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch Kernel
    rgbToGrayKernel<<<gridSize, blockSize>>>(d_rgbImage, d_grayImage, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(outputImage.ptr<unsigned char>(), d_grayImage, graySize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_rgbImage);
    cudaFree(d_grayImage);
}

int main() {
    // Load input image
    Mat inputImage = imread("rgb.jpg"); // Change to your image path
    if (inputImage.empty()) {
        cout << "Error: Image not found!" << endl;
        return -1;
    }

    // Convert image to uchar3 format
    Mat rgbImage;
    cvtColor(inputImage, rgbImage, COLOR_BGR2RGB);

    // Create output grayscale image
    Mat grayImage(rgbImage.rows, rgbImage.cols, CV_8UC1);

    // Convert using CUDA
    convertRGBToGrayCUDA(rgbImage, grayImage);

    // Save and display
    imwrite("gray_output.jpg", grayImage);
    imshow("Grayscale Image", grayImage);
    waitKey(0);

    return 0;
}

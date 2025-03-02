#include <iostream>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found!" << std::endl;
        return -1;
    }

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "======================================" << std::endl;
        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Multiprocessors (SMs): " << prop.multiProcessorCount << std::endl;
        std::cout << "CUDA Cores per SM: " << (prop.major >= 3 ? prop.multiProcessorCount * 128 : prop.multiProcessorCount * 8) << std::endl;
        std::cout << "Total CUDA Cores: " << prop.multiProcessorCount * (prop.major >= 3 ? 128 : 8) << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "Max Grid Size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "Max Block Dimensions: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}

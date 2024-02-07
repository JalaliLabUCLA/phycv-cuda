#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ cufftComplex cexpf(cufftComplex z); 
__device__ float atomicMaxf(float* address, float val); 
__device__ float atomicMinf(float* address, float val);
__global__ void BGR2HSVKernel(uint8_t* image, int width, int height, int step); 
__global__ void HSV2BGRKernel(uint8_t* image, int width, int height, int step); 
__global__ void init_kernel(cufftComplex* d_vevid_kernel, float S, float T, int width, int height);
__global__ void max_reduce(const cufftComplex* const d_array, float* d_max, const size_t N);
__global__ void min_max_reduce(cufftComplex* d_array, float* d_max, float* d_min, const size_t N);
__global__ void scale_exp(cufftComplex* d_vevid_kernel, float scalar, const size_t N);
__global__ void scale(cufftComplex* input, float scalar, const size_t N);
__global__ void fftshift(cufftComplex* d_vevid_kernel, const size_t width, const size_t height);
__global__ void populate(cufftComplex* d_image, uint8_t* d_buffer, float b, const size_t N);
__global__ void hadamard(cufftComplex* a1, cufftComplex* a2, const size_t N);
__global__ void phase(cufftComplex* vevid_image, uint8_t* image, float gain, const size_t N);
__global__ void vevid_phase(cufftComplex* vevid_image, uint8_t* image, float gain, const size_t N);
__global__ void norm(cufftComplex* d_image, uint8_t* d_buffer, float max_phase, float min_phase, int N);

#endif // KERNELS_CUH
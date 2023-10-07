#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

__device__ float atomicMaxf(float* address, float val);
__device__ float atomicMinf(float* address, float val);
__global__ void min_reduce(const cufftComplex* const d_array, float* d_max, const size_t N);
__global__ void max_reduce(const cufftComplex* const d_array, float* d_max, const size_t N);
__global__ void vevid_kernel(cufftComplex* array, float phase_strength, float variance, const size_t width, const size_t height);
__global__ void vevid_phase(cufftComplex* vevid_image, uint8_t* image, float gain, const size_t N);
__global__ void vevid_normalize(cufftComplex* vevid_image, uint8_t* image, float max_phase, float min_phase, const size_t N);
__global__ void populate_real(cufftComplex* d_image, uint8_t* d_buffer, const size_t N);
__global__ void fftshift(cufftComplex* data, const size_t width, const size_t height);
__global__ void hadamard(cufftComplex* a1, cufftComplex* a2, const size_t N);
__global__ void add(cufftComplex* input, float addend, const size_t N);
__global__ void scale(cufftComplex* input, float scalar, const size_t N);
__global__ void arg(cufftComplex* input, uint8_t* output, const size_t N);
__global__ void vevid_phase_lite(uint8_t* input, float gain, float regularization_term, const size_t N);

void convert_to_hsv_wrapper(uchar3 *rgb, float3 *hsv, int width, int height); 

#endif

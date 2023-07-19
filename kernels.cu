#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <stdint.h>
#include <float.h>



// CUDA kernel implementations
__device__ 
float atomicMaxf(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(val));
    }
    return __int_as_float(old);
}

__device__ 
float atomicMinf(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(val));
    }
    return __int_as_float(old);
}

__global__
void min_reduce(const cufftComplex* const d_array, float* d_max, const size_t N)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = FLT_MAX;

    while (gid < N) {
        shared[tid] = min(shared[tid], d_array[gid].x);
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && gid < N)
            shared[tid] = min(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMinf(d_max, shared[0]);
}

__global__ 
void max_reduce(const cufftComplex* const d_array, float* d_max, const size_t N)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = FLT_MIN;

    while (gid < N) {
        shared[tid] = max(shared[tid], d_array[gid].x);
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && gid < N)
            shared[tid] = max(shared[tid], shared[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
        atomicMaxf(d_max, shared[0]);
}

__global__
void vevid_kernel(cufftComplex* array, float phase_strength, float variance, const size_t width, const size_t height)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < width * height; i += stride) {
        int row = i / width;
        int col = i % width;
        float u = -0.5 + ((0.5 + 0.5) / (height - 1)) * floorf((row * width + col) / width);
        float v = -0.5 + ((0.5 + 0.5) / (width - 1)) * col;
        float value = sqrtf((u * u) + (v * v));
        float x = expf(-(value * value) / variance);
        x = x / phase_strength;
        array[i].x = x; 
    }
}

__global__
void vevid_phase(cufftComplex* vevid_image, uint8_t* image, float gain, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float max = M_PI / 2;
    float min = -M_PI / 2;
    for (int i = index; i < N; i += stride) {
        float imaginary = gain * cuCimagf(vevid_image[i]);
        float original = (float)image[i];
        float temp = atan2f(imaginary, original);
        temp = (temp - min) / (max - min);
        image[i] = static_cast<uint8_t>(temp * 255);
    }
}

__global__
void populate_real(cufftComplex* d_image, uint8_t* d_buffer, const size_t N) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        float temp = d_buffer[i];
        d_image[i].x = temp / 255.0f;
    }
}

__global__
void fftshift(cufftComplex* data, const size_t width, const size_t height) 
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < width * height; i += stride) {
        if (i >= (width * height) / 2) {
            return;
        }
        else {
            if (i % width < (width / 2)) {
                cufftComplex temp = data[i];
                data[i] = data[i + (width * (height / 2)) + (width / 2)];
                data[i + (width * (height / 2)) + (width / 2)] = temp;
            }
            else {
                cufftComplex temp = data[i];
                data[i] = data[i + (width * (height / 2)) - (width / 2)];
                data[i + (width * (height / 2)) - (width / 2)] = temp;
            }
        }

    }
}

__global__
void hadamard(cufftComplex* a1, cufftComplex* a2, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    cufftComplex temp;
    for (int i = index; i < N; i += stride) {
        float x = a1[i].x; 
        a1[i].x = cosf(-x); 
        a1[i].y = sinf(-x); 
        temp.x = a1[i].x * a2[i].x - a1[i].y * a2[i].y;
        temp.y = a1[i].x * a2[i].y + a1[i].y * a2[i].x;
        a2[i].x = temp.x;
        a2[i].y = temp.y;
    }

}

__global__
void add(cufftComplex* input, float addend, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        input[i].x += addend;
    }

}

__global__
void scale(cufftComplex* input, float scalar, const size_t N) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        input[i].x = input[i].x * scalar;
        input[i].y = input[i].y * scalar;
    }
}

__global__
void arg(cufftComplex* input, uint8_t* output, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        float temp = atan2f(cuCimagf(input[i]), cuCrealf(input[i]));
        output[i] = static_cast<uint8_t>(temp * 255);
    }
}
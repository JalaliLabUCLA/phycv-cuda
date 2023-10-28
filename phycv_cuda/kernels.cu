#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <stdint.h>
#include <float.h>
#include <stdio.h>

__global__
void init_kernel(cufftComplex* d_vevid_kernel, float S, float T, int width, int height)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < width * height; i += stride) {
        int row = i / width;
        int col = i % width;
        float u = -0.5 + ((0.5 + 0.5) / (height - 1)) * floorf((row * width + col) / width);
        float v = -0.5 + ((0.5 + 0.5) / (width - 1)) * col;
        float value = sqrtf((u * u) + (v * v));
        float x = expf(-(value * value) / T);
        d_vevid_kernel[i].x = x; 
    }
}

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
void max_reduce(const cufftComplex* const d_array, float* d_max, const size_t N)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLT_MAX;

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
void min_max_reduce(cufftComplex* d_array, float* d_max, float* d_min, const size_t N)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;
    shared[tid] = -FLT_MAX;
    shared[tid + blockDim.x] = FLT_MAX;

    while (gid < N) {
        shared[tid] = max(shared[tid], d_array[gid].x);
        shared[tid + blockDim.x] = min(shared[tid + blockDim.x], d_array[gid].x);
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();
    gid = (blockDim.x * blockIdx.x) + tid;  

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && gid < N) {
            shared[tid] = max(shared[tid], shared[tid + s]);
            shared[tid + blockDim.x] = min(shared[tid + blockDim.x], shared[tid + blockDim.x + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxf(d_max, shared[0]);
        atomicMinf(d_min, shared[blockDim.x]);
    }
}

__device__ __forceinline__
cufftComplex cexpf(cufftComplex z)
{
    cufftComplex res; 
    float t = expf(z.x); 
    sincosf(z.y, &res.y, &res.x); 
    res.x *= t; 
    res.y *= t; 
    return res; 
}

__global__
void scale_exp(cufftComplex* d_vevid_kernel, float scalar, const size_t N) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        float original = d_vevid_kernel[i].x; 
        original = original * scalar; 

        cufftComplex temp; 
        temp.x = 0; 
        temp.y = -original; 
        
        d_vevid_kernel[i] = cexpf(temp); 
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
void fftshift(cufftComplex* d_vevid_kernel, const size_t width, const size_t height) 
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < width * height; i += stride) {
        if (i >= (width * height) / 2) {
            return;
        }
        else {
            if (i % width < (width / 2)) {
                cufftComplex temp = d_vevid_kernel[i];
                d_vevid_kernel[i] = d_vevid_kernel[i + (width * (height / 2)) + (width / 2)];
                d_vevid_kernel[i + (width * (height / 2)) + (width / 2)] = temp;
            }
            else {
                cufftComplex temp = d_vevid_kernel[i];
                d_vevid_kernel[i] = d_vevid_kernel[i + (width * (height / 2)) - (width / 2)];
                d_vevid_kernel[i + (width * (height / 2)) - (width / 2)] = temp;
            }
        }
    }
}

__global__
void populate(cufftComplex* d_image, uint8_t* d_buffer, float b, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        float temp = d_buffer[i];
        temp = (temp / 255.0f) + b;
        d_image[i].x = temp; 
        d_image[i].y = 0.0f; // BUG FIX HERE -- MUST RESET COMPLEX VALUE TO 0
    }
}

__global__
void hadamard(cufftComplex* a1, cufftComplex* a2, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;


    for (int i = index; i < N; i += stride) {
        cufftComplex a1_val = a1[i];
        cufftComplex a2_val = a2[i];
        
        // Perform complex componentwise multiplication
        cufftComplex result;
        result.x = a1_val.x * a2_val.x - a1_val.y * a2_val.y;
        result.y = a1_val.x * a2_val.y + a1_val.y * a2_val.x;

        // Store the result back in a2
        a2[i] = result;
    }
}

__global__
void phase(cufftComplex* vevid_image, uint8_t* image, float gain, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        float imaginary = gain * cuCimagf(vevid_image[i]);
        float original = (float)image[i] / 255.0f;
        float temp = atan2f(imaginary, original);
        vevid_image[i].x = temp; 
    }
}

__global__
void norm(cufftComplex* d_image, uint8_t* d_buffer, float max_phase, float min_phase, int N) // TODO: change function signature to expect float* 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 
    for (int i = index; i < N; i += stride) {
        float temp = d_image[i].x; 
        temp = ((temp - min_phase) / (max_phase - min_phase)); 
        d_buffer[i] = static_cast<uint8_t>(temp * 255);
    }
}
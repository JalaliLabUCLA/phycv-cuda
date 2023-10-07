#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <stdint.h>
#include <float.h>



// CUDA kernel implementations

__device__ 
uchar3 convert_one_pixel_to_rgb(float3 pixel) {
    float r, g, b;
    float h, s, v;

    h = pixel.x;
    s = pixel.y;
    v = pixel.z;

    float f = h / 60.0f;
    float hi = floorf(f);
    f = f - hi;
    float p = v * (1 - s);
    float q = v * (1 - s * f);
    float t = v * (1 - s * (1 - f));

    if (hi == 0.0f || hi == 6.0f) {
        r = v;
        g = t;
        b = p;
    } else if (hi == 1.0f) {
        r = q;
        g = v;
        b = p;
    } else if (hi == 2.0f) {
        r = p;
        g = v;
        b = t;
    } else if (hi == 3.0f) {
        r = p;
        g = q;
        b = v;
    } else if (hi == 4.0f) {
        r = t;
        g = p;
        b = v;
    } else {
        r = v;
        g = p;
        b = q;
    }

    unsigned char red = (unsigned char)__float2uint_rn(255.0f * r);
    unsigned char green = (unsigned char)__float2uint_rn(255.0f * g);
    unsigned char blue = (unsigned char)__float2uint_rn(255.0f * b);

    return make_uchar3(red, green, blue);
}

__device__ 
float3 convert_one_pixel_to_hsv(uchar3 pixel) {
    float r, g, b;
    float h, s, v;

    r = pixel.x / 255.0f;
    g = pixel.y / 255.0f;
    b = pixel.z / 255.0f;

    float max = fmax(r, fmax(g, b));
    float min = fmin(r, fmin(g, b));
    float diff = max - min;

    v = max;

    if (v == 0.0f) { // black
        h = s = 0.0f;
    } else {
        s = diff / v;
        if (diff < 0.001f) { // grey
            h = 0.0f;
        } else { // color
            if (max == r) {
                h = 60.0f * (g - b) / diff;
                if (h < 0.0f) { h += 360.0f; }
            } else if (max == g) {
                h = 60.0f * (2 + (b - r) / diff);
            } else {
                h = 60.0f * (4 + (r - g) / diff);
            }
        }
    }

    return make_float3(h, s, v);
}

__global__ 
void convert_to_hsv(uchar3* rgb, float3* hsv, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        uchar3 rgb_pixel = rgb[x + width * y];
        float3 hsv_pixel = convert_one_pixel_to_hsv(make_uchar3(rgb_pixel.x, rgb_pixel.y, rgb_pixel.z));
        hsv[x + width * y] = hsv_pixel;
    }
}

void convert_to_hsv_wrapper(uchar3 *rgb, float3 *hsv, int width, int height) {
	dim3 threads(16,16);
	dim3 blocks((width + 15)/16, (height + 15)/16);
	
	convert_to_hsv<<<blocks, threads>>>(rgb, hsv, width, height);
}

/*
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
*/

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
        x = x * phase_strength;
        array[i].x = x; 
    }
}

// TODO: Compute min and max from the actual input. 
__global__
void vevid_phase(cufftComplex* vevid_image, uint8_t* image, float gain, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float max = 0;
    float min = -M_PI / 2;
    for (int i = index; i < N; i += stride) {
        float imaginary = gain * cuCimagf(vevid_image[i]);
        float original = (float)image[i];
        float temp = atan2f(imaginary, original);
        vevid_image[i].x = temp; 
        temp = (temp - min) / (max - min);
        image[i] = static_cast<uint8_t>(temp * 255);
    }
}

__global__
void vevid_normalize(cufftComplex* vevid_image, uint8_t* image, float max_phase, float min_phase, const size_t N) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 
    for (int i = index; i < N; i += stride) {
        float temp = vevid_image[i].x; 
        temp = (temp - min_phase) / (max_phase - min_phase); 
        image[i] = static_cast<uint8_t>(temp * 255); 
    }
}

// TODO: Compute min and max from actual input. 

__global__
void vevid_phase_lite(uint8_t* input, float gain, float regularization_term, const size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; 
    float max = M_PI / 2; 
    float min = -M_PI / 2; 
    for (int i = index; i < N; i+= stride) {
        float imaginary = -gain * (input[i] + regularization_term);
        float real = input[i]; 
        float temp = atan2f(imaginary, real); 
        temp = (temp - min) / (max - min); 
        input[i] = static_cast<uint8_t>(temp * 1000);
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

/*
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
*/

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

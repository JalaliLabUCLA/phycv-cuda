#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <npp.h>

#include <iostream>
#include <string>
#include <vector> 
#include <chrono>

#include "vevid.cuh"
#include "kernels.cuh"

using namespace cv; 
using namespace std;

// TODO: change to C++ class style. 
// TODO: look into how timing scales for object detection with different resolutions.

/* Timing Macros */
#define MEASURE_GPU_TIME(func, result)                                                      \
    do                                                                                      \
    {                                                                                       \
        cudaEvent_t startEvent, stopEvent;                                                  \
        cudaEventCreate(&startEvent);                                                       \
        cudaEventCreate(&stopEvent);                                                        \
        cudaEventRecord(startEvent);                                                        \
        func;                                                                               \
        cudaEventRecord(stopEvent);                                                         \
        cudaEventSynchronize(stopEvent);                                                    \
        float milliseconds = 0;                                                             \
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);                         \
        result += static_cast<float>(milliseconds);                                         \
        cudaEventDestroy(startEvent);                                                       \
        cudaEventDestroy(stopEvent);                                                        \
    } while (0)

#define MEASURE_CPU_TIME(func, result)                                                      \
    do                                                                                      \
    {                                                                                       \
        auto start = chrono::high_resolution_clock::now();                                  \
        func;                                                                               \
        auto stop = std::chrono::high_resolution_clock::now();                              \
        chrono::duration<float, milli> elapsed = stop - start;                              \
        result = static_cast<double>(elapsed.count());                                      \
    } while (0)


struct Parameters { // TODO: change to b,G,S,T, explain meaning once
    float phase_strength; 
    float warp_strength; 
    float spectral_phase_variance;
    float regularization_term; 
    float phase_activation_gain; 

    Parameters() : phase_strength(1), warp_strength(1), spectral_phase_variance(1), 
        regularization_term(1), phase_activation_gain(1)
    {}
};

struct Times {
    float BGR_to_HSV_time; 
    float data_in_time; 
    float fft_time; 
    float kernel_multiplication_time; 
    float ifft_time; 
    float phase_time;
    float data_out_time; 
    float HSV_to_BGR_time;  
    float vevid_time; 
}; 

struct VevidContext
{
    /* GPU resources */
    cufftComplex *d_vevid_kernel; 
    cufftComplex *d_image;
    uint8_t *d_buffer; 
    float *d_max; 
    float *d_min;
    float *d_phase_min; 
    float *d_phase_max; 

    uchar3* d_rgb; 
    float3* d_hsv; 

    /* FFT plan */ 
    cufftHandle plan; 

    /* Input parameters */
    Parameters params; 

    /* Frame width and height */
    size_t width; 
    size_t height; 
}; 

VevidContext context; 
Times times; 

void vevid_init(int width, int height,
    float phase_strength, 
    float spectral_phase_variance, 
    float regularization_term, 
    float phase_activation_gain) 
{
    context.width = width; 
    context.height = height; 
    const size_t N = width * height; 

    /* Allocate GPU memory */
    cudaMalloc((void**)&context.d_vevid_kernel, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&context.d_image, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&context.d_buffer, N * sizeof(uint8_t)); 
    cudaMalloc((void**)&context.d_max, sizeof(float)); 
    cudaMalloc((void**)&context.d_min, sizeof(float)); 
    cudaMalloc((void**)&context.d_phase_max, sizeof(float)); 
    cudaMalloc((void**)&context.d_phase_min, sizeof(float)); 

    cudaMalloc((void**)&context.d_rgb, N * sizeof(uchar3)); 
    cudaMalloc((void**)&context.d_hsv, N * sizeof(float3)); 

    /* Initialize FFT plan */
    cufftPlan2d(&context.plan, (int)height, (int)width, CUFFT_C2C); 

    /* Set up input parameters */
    context.params.phase_strength = phase_strength; 
    context.params.spectral_phase_variance = spectral_phase_variance; 
    context.params.regularization_term = regularization_term; 
    context.params.phase_activation_gain = phase_activation_gain; 

    int block_size = 32; 
    int grid_size = ((int)N + block_size - 1) / block_size; 

    /* Compute VEViD kernel */
    vevid_kernel<<<grid_size, block_size>>>(context.d_vevid_kernel, context.params.phase_strength, context.params.spectral_phase_variance, width, height);
    max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_vevid_kernel, context.d_max, N);
    float max_val; 
    cudaMemcpy(&max_val, context.d_max, sizeof(float), cudaMemcpyDeviceToHost);  
    scale<<<grid_size, block_size>>>(context.d_vevid_kernel, (1.0f / max_val), N);
    fftshift<<<grid_size, block_size>>>(context.d_vevid_kernel, width, height);
}

void vevid(cv::Mat& image, bool show_timing, bool lite) {
    auto vevid_start = chrono::high_resolution_clock::now(); 

    size_t width = context.width; 
    size_t height = context.height; 
    size_t N = width * height; 

    /* Convert from BGR to HSV */
    float to_HSV_time; 
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_BGR2HSV), to_HSV_time);
    
    /*
    cvtColor(image, image, COLOR_BGR2RGB); 
    uchar3* rgb = image.ptr<uchar3>(0); 
    uchar3* out; 
    cudaMemcpy(context.d_rgb, rgb, N * sizeof(uchar3), cudaMemcpyHostToDevice); 
    convert_to_hsv_wrapper(context.d_rgb, context.d_hsv, width, height); 
    */

    /* Get pointer to V channel of HSV matrix */
    auto start = chrono::high_resolution_clock::now(); 
    vector<Mat> hsv_channels; 
    split(image, hsv_channels); 
    uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0); 
    auto stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, std::milli> elapsed = stop - start; 
    times.BGR_to_HSV_time = to_HSV_time + elapsed.count(); 

    /* -- Start of Algorithm Code -- */
    /* Copy data from host to device */
    MEASURE_GPU_TIME(cudaMemcpy(context.d_buffer, idata, N * sizeof(uint8_t), cudaMemcpyHostToDevice), times.data_in_time); 

    /* Call CUDA kernels */
    int block_size = 32; 
    int grid_size = ((int)N + block_size - 1) / block_size; 

    if (lite) {
        vevid_phase_lite<<<grid_size, block_size>>>(context.d_buffer, context.params.phase_activation_gain, context.params.regularization_term, N); 
    }
    else {
        /* Take FFT */
        float populate_real_time; 
        float add_time; 
        float forward_fft_time; 
        MEASURE_GPU_TIME((populate_real<<<grid_size, block_size>>>(context.d_image, context.d_buffer, N)), populate_real_time);
        MEASURE_GPU_TIME((add<<<grid_size, block_size>>>(context.d_image, context.params.regularization_term, N)), add_time);
        MEASURE_GPU_TIME(cufftExecC2C(context.plan, context.d_image, context.d_image, CUFFT_FORWARD), forward_fft_time);
        times.fft_time = populate_real_time + add_time + forward_fft_time;

        /* Multiply kernel with image in frequency domain */
        float hadamard_time; 
        MEASURE_GPU_TIME((hadamard<<<grid_size, block_size>>>(context.d_vevid_kernel, context.d_image, N)), hadamard_time);
        times.kernel_multiplication_time = hadamard_time; 

        /* Take IFFT */
        float backward_fft_time; 
        float scale_time; 
        MEASURE_GPU_TIME(cufftExecC2C(context.plan, context.d_image, context.d_image, CUFFT_INVERSE), backward_fft_time);
        MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(context.d_image, (1.0f / (float)N), N)), scale_time);
        times.ifft_time = backward_fft_time + scale_time; 

        /* Get phase */
        float phase_time; 
        MEASURE_GPU_TIME((vevid_phase<<<grid_size, block_size>>>(context.d_image, context.d_buffer, context.params.phase_activation_gain, N)), phase_time);
        //max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_image, context.d_phase_max, N);
        //min_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_image, context.d_phase_min, N);
        //float max_phase;
        //float min_phase;  
        //cudaMemcpy(&max_phase, context.d_phase_max, sizeof(float), cudaMemcpyDeviceToHost); 
        //cudaMemcpy(&min_phase, context.d_phase_min, sizeof(float), cudaMemcpyDeviceToHost); 
        //vevid_normalize<<<grid_size, block_size>>>(context.d_image, context.d_buffer, 0, -M_PI / 2, N);
        //cout << max_phase << ", " << min_phase << endl; 
        times.phase_time = phase_time; 
    }

    /* Copy data from device to host */
    MEASURE_GPU_TIME(cudaMemcpy(idata, context.d_buffer, N * sizeof(uint8_t), cudaMemcpyDeviceToHost), times.data_out_time); 
    /* -- End of Algorithm Code -- */

    /* Merge channels */
    float merge_time; 
    float to_BGR_time; 
    MEASURE_CPU_TIME(merge(hsv_channels, image), merge_time); 

    /* Normalize image */
    //cv::normalize(image, image, 0, 255, NORM_MINMAX); 

    /* Convert from HSV to BGR */
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_HSV2BGR), to_BGR_time);
    times.HSV_to_BGR_time = merge_time + to_BGR_time; 

    auto vevid_stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, milli> total = vevid_stop - vevid_start; 
    times.vevid_time = total.count(); 

    if (show_timing) {
        cout << "Timing Results: " << endl; 
        cout << "BGR to HSV time per frame: " << times.BGR_to_HSV_time << " ms" << endl; 
        cout << "Forward data copy time: " << times.data_in_time << " ms" << endl; 
        cout << "FFT time per frame: " << times.fft_time << " ms" << endl; 
        cout << "Kernel multiplication time per frame: " << times.kernel_multiplication_time << " ms" << endl; 
        cout << "Inverse FFT time per frame: " << times.ifft_time << " ms" << endl; 
        cout << "Phase time per frame: " << times.phase_time << " ms" << endl; 
        cout << "Backward data copy time per frame: " << times.data_out_time << " ms" << endl; 
        cout << "HSV to BGR time per frame: " << times.HSV_to_BGR_time << " ms" << endl; 
        cout << "Total time per frame: " << times.vevid_time << " ms" << endl; 
        cout << endl; 
    }
}

void vevid_fini() {
    /* Free GPU memory */
    cudaFree(context.d_vevid_kernel); 
    cudaFree(context.d_image); 
    cudaFree(context.d_buffer); 
    cudaFree(context.d_max); 
    cudaFree(context.d_min); 

    /* Destroy FFT plan */
    cufftDestroy(context.plan); 
}

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
    }
}

__global__
void hadamard1(cufftComplex* a1, cufftComplex* a2, const size_t N)
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

Vevid::Vevid(int width, int height, float S, float T, float b, float G)
: m_width(width), m_height(height), m_S(S), m_T(T), m_b(b), m_G(G), d_vevid_kernel(nullptr), d_image(nullptr), d_buffer(nullptr), 
t_BGRtoHSV(0)
{
    cudaMalloc((void**)&d_vevid_kernel, m_width * m_height * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_image, m_width * m_height * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_buffer, m_width * m_height * sizeof(uint8_t)); 
    cufftPlan2d(&m_plan, height, width, CUFFT_C2C); 

    int block_size = 32; 
    int grid_size = ((width * height) + block_size - 1) / block_size; 

    float* d_max; 
    float max_val; 
    init_kernel<<<grid_size, block_size>>>(d_vevid_kernel, S, T, width, height); 
    cudaMalloc((void**)&d_max, sizeof(float)); 
    max_reduce<<<64, block_size, block_size * sizeof(float)>>>(d_vevid_kernel, d_max, width * height);
    cudaMemcpy(&max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost);  
    scale_exp<<<grid_size, block_size>>>(d_vevid_kernel, (S / max_val), width * height);
    fftshift<<<grid_size, block_size>>>(d_vevid_kernel, width, height);

    /*
    cufftComplex* temp = new cufftComplex[m_width * m_height];
    cudaMemcpy(temp, d_vevid_kernel, width * height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << "(" << temp[i * width + j].x << ", " << temp[i * width + j].y << ")"; 
        }
        cout << endl; 
    }

    delete[] temp; 
    */
}

void Vevid::run(Mat& image, bool show_timing, bool lite)
{
    // Convert from BGR to HSV and get V channel of image
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_BGR2HSV), t_BGRtoHSV);
    auto start = chrono::high_resolution_clock::now(); 
    vector<Mat> hsv_channels; 
    split(image, hsv_channels); 
    uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0); 
    auto stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, std::milli> elapsed = stop - start; 
    t_BGRtoHSV += elapsed.count(); 

    MEASURE_GPU_TIME(cudaMemcpy(d_buffer, idata, m_width * m_height * sizeof(uint8_t), cudaMemcpyHostToDevice), t_iCopy); 

    // Call CUDA kernels
    int block_size = 32; 
    int grid_size = ((m_width * m_height) + block_size - 1) / block_size; 

    // Take FFT
    MEASURE_GPU_TIME((populate<<<grid_size, block_size>>>(d_image, d_buffer, m_b, m_width * m_height)), t_populate);
    MEASURE_GPU_TIME(cufftExecC2C(m_plan, d_image, d_image, CUFFT_FORWARD), t_fft); 
    //MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(d_image, (1.0f / (m_width * m_height)), m_width * m_height)), t_ifft);

    // Multiply kernel with image in frequency domain
    MEASURE_GPU_TIME((hadamard1<<<grid_size, block_size>>>(d_vevid_kernel, d_image, m_width * m_height)), t_hadamard);

    // Take IFFT
    MEASURE_GPU_TIME(cufftExecC2C(m_plan, d_image, d_image, CUFFT_INVERSE), t_ifft);
    MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(d_image, (1.0f / (m_width * m_height)), m_width * m_height)), t_ifft);
    
    // Get phase
    MEASURE_GPU_TIME((vevid_phase<<<grid_size, block_size>>>(d_image, d_buffer, m_G, m_width * m_height)), t_phase);
    //vevid_normalize<<<grid_size, block_size>>>(d_image, d_buffer, 0, -M_PI / 2, m_width * m_height);

    MEASURE_GPU_TIME(cudaMemcpy(idata, d_buffer, m_width * m_height * sizeof(uint8_t), cudaMemcpyDeviceToHost), t_oCopy); 

    // Merge channels and convert from HSV to BGR
    MEASURE_CPU_TIME(merge(hsv_channels, image), t_HSVtoBGR); 
    //cv::normalize(image, image, 0, 255, NORM_MINMAX);
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_HSV2BGR), t_HSVtoBGR);

    /*
    cufftComplex* temp = new cufftComplex[m_width * m_height];
    cudaMemcpy(temp, d_image, m_width * m_height * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m_height; i++) {
        for (int j = 0; j < m_width; j++) {
            cout << "(" << temp[i * m_width + j].x << ", " << temp[i * m_width + j].y << ")"; 
        }
        cout << endl; 
    }

    delete[] temp; 
    */
}

Vevid::~Vevid()
{

}





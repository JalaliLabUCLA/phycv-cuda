#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <iostream>
#include <string>
#include <vector> 
#include <chrono>

#include "vevid.cuh"
#include "kernels.cuh"

using namespace cv; 
using namespace std;

/* Timing Macros */
#define MEASURE_GPU_TIME(func, result)                        \
    do                                                        \
    {                                                         \
        cudaEvent_t startEvent, stopEvent;                    \
        cudaEventCreate(&startEvent);                         \
        cudaEventCreate(&stopEvent);                          \
        cudaEventRecord(startEvent);                          \
        func;                                                 \
        cudaEventRecord(stopEvent);                           \
        cudaEventSynchronize(stopEvent);                      \
        float milliseconds = 0;                               \
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent); \
        result = static_cast<double>(milliseconds);           \
        cudaEventDestroy(startEvent);                         \
        cudaEventDestroy(stopEvent);                          \
    } while (0)

struct Parameters {
    float phase_strength; 
    float warp_strength; 
    float spectral_phase_variance;
    float regularization_term; 
    float phase_activation_gain; 

    Parameters() : phase_strength(1), warp_strength(1), spectral_phase_variance(1), 
        regularization_term(1), phase_activation_gain(1)
    {}
};

struct VevidContext
{
    /* GPU resources */
    cufftComplex *d_vevid_kernel; 
    cufftComplex *d_image;
    uint8_t *d_buffer; 
    float *d_max; 
    float *d_min; 

    /* FFT plan */ 
    cufftHandle plan; 

    /* Input parameters */
    Parameters params; 

    /* Frame width and height */
    size_t width; 
    size_t height; 
}; 

VevidContext context; 

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

    /* Initialize FFT plan */
    cufftPlan2d(&context.plan, (int)height, (int)width, CUFFT_C2C); 

    /* Set up input parameters */
    context.params.phase_strength = phase_strength; 
    context.params.spectral_phase_variance = spectral_phase_variance; 
    context.params.regularization_term = regularization_term; 
    context.params.phase_activation_gain = phase_activation_gain; 
}

void vevid(cv::Mat& image) {
    size_t width = context.width; 
    size_t height = context.height; 
    size_t N = width * height; 

    /* Convert from BGR to HSV */
    cvtColor(image, image, COLOR_BGR2HSV); 

    /* Get pointer to V channel of HSV matrix */
    vector<Mat> hsv_channels; 
    split(image, hsv_channels); 
    uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0); 

    /* -- Start of Algorithm Code -- */
    /* Copy data from host to device */
    cudaMemcpy(context.d_buffer, idata, N * sizeof(uint8_t), cudaMemcpyHostToDevice); 

    /* Call CUDA kernels */
    int block_size = 32; 
    int grid_size = ((int)N + block_size - 1) / block_size; 

    /* Take FFT */
    populate_real<<<grid_size, block_size>>>(context.d_image, context.d_buffer, N);
    add<<<grid_size, block_size>>>(context.d_image, context.params.regularization_term, N);
    cufftExecC2C(context.plan, context.d_image, context.d_image, CUFFT_FORWARD); 

    /* Compute VEViD kernel */
    vevid_kernel<<<grid_size, block_size>>>(context.d_vevid_kernel, context.params.phase_strength, context.params.spectral_phase_variance, width, height);
    max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_vevid_kernel, context.d_max, N);
    float max_val; 
    cudaMemcpy(&max_val, context.d_max, sizeof(float), cudaMemcpyDeviceToHost); 
    scale<<<grid_size, block_size>>>(context.d_vevid_kernel, (1.0f / max_val), N);
    fftshift<<<grid_size, block_size>>>(context.d_vevid_kernel, width, height);

    /* Multiply kernel with image in frequency domain */
    hadamard<<<grid_size, block_size>>>(context.d_vevid_kernel, context.d_image, N);

    /* Take IFFT */
    cufftExecC2C(context.plan, context.d_image, context.d_image, CUFFT_INVERSE);
    scale<<<grid_size, block_size>>>(context.d_image, (1.0f / (float)N), N);

    /* Get phase */
    vevid_phase<<<grid_size, block_size>>>(context.d_image, context.d_buffer, context.params.phase_activation_gain, N); 

    /* Copy data from device to host */
    cudaMemcpy(idata, context.d_buffer, N * sizeof(uint8_t), cudaMemcpyDeviceToHost); 
    /* -- End of Algorithm Code -- */

    /* Merge channels */
    merge(hsv_channels, image); 

    /* Convert from HSV to BGR */
    cvtColor(image, image, COLOR_HSV2BGR); 

    //cudaDeviceSynchronize();
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

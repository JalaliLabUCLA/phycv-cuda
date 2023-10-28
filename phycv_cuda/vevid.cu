#include <iostream>
#include <string>
#include <vector> 
#include <chrono>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

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


Vevid::Vevid(int width, int height, float S, float T, float b, float G)
: m_width(width), m_height(height), m_S(S), m_T(T), m_b(b), m_G(G), d_vevid_kernel(nullptr), d_image(nullptr), d_buffer(nullptr), d_max_phase(nullptr), d_min_phase(nullptr),
t_BGRtoHSV(0), t_iCopy(0), t_fft(0), t_hadamard(0), t_ifft(0), t_phase(0), t_oCopy(0), t_HSVtoBGR(0)
{
    const int N = m_width * m_height; 

    cudaMalloc((void**)&d_vevid_kernel, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_image, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_buffer, N * sizeof(uint8_t)); 
    cufftPlan2d(&m_plan, m_height, m_width, CUFFT_C2C); 

    cudaMalloc((void**)&d_max_phase, sizeof(float)); // TODO: void** needed?
    cudaMalloc((void**)&d_min_phase, sizeof(float)); 
    cudaMemset(d_max_phase, -100, sizeof(float)); // less than PI / 2
    cudaMemset(d_min_phase, 100, sizeof(float)); // greater than PI / 2

    int block_size = 32; 
    int grid_size = ((N) + block_size - 1) / block_size; 

    float* d_max; 
    float max_val;
    cudaMalloc((void**)&d_max, sizeof(float));  
    init_kernel<<<grid_size, block_size>>>(d_vevid_kernel, S, T, m_width, m_height); 
    max_reduce<<<64, block_size, block_size * sizeof(float)>>>(d_vevid_kernel, d_max, N);
    cudaMemcpy(&max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost);                 //TODO: change to gpu values
    scale_exp<<<grid_size, block_size>>>(d_vevid_kernel, (S / max_val), N);
    fftshift<<<grid_size, block_size>>>(d_vevid_kernel, m_width, m_height);
    cudaFree(d_max); 
}


void Vevid::run(cv::Mat& image, bool show_timing, bool lite)
{
    auto vevid_start = chrono::high_resolution_clock::now(); 
    const int N = m_width * m_height; 
    
    // Convert from BGR to HSV and get V channel of image
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_BGR2HSV), t_BGRtoHSV);
    auto start = chrono::high_resolution_clock::now(); 
    vector<Mat> hsv_channels; 
    split(image, hsv_channels); 
    uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0); 
    auto stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, std::milli> elapsed = stop - start; 
    t_BGRtoHSV += elapsed.count(); 

    MEASURE_GPU_TIME(cudaMemcpy(d_buffer, idata, N * sizeof(uint8_t), cudaMemcpyHostToDevice), t_iCopy); 

    // Call CUDA kernels
    int block_size = 32; 
    int grid_size = (N + block_size - 1) / block_size; 

    // Take FFT
    MEASURE_GPU_TIME((populate<<<grid_size, block_size>>>(d_image, d_buffer, m_b, N)), t_fft);
    MEASURE_GPU_TIME(cufftExecC2C(m_plan, d_image, d_image, CUFFT_FORWARD), t_fft); 

    // Multiply kernel with image in frequency domain
    MEASURE_GPU_TIME((hadamard<<<grid_size, block_size>>>(d_vevid_kernel, d_image, N)), t_hadamard);

    // Take IFFT
    MEASURE_GPU_TIME(cufftExecC2C(m_plan, d_image, d_image, CUFFT_INVERSE), t_ifft);
    MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(d_image, (1.0f / (N)), N)), t_ifft);
    
    // Get phase
    MEASURE_GPU_TIME((phase<<<grid_size, block_size>>>(d_image, d_buffer, m_G, N)), t_phase);

    MEASURE_GPU_TIME(cudaMemset(d_max_phase, -10, sizeof(float)), t_phase); // less than PI / 2
    MEASURE_GPU_TIME(cudaMemset(d_min_phase, 10, sizeof(float)), t_phase); // greater than PI / 2
    MEASURE_GPU_TIME((min_max_reduce<<<64, block_size, block_size * sizeof(float)>>>(d_image, d_max_phase, d_min_phase, N)), t_phase);
    float max_val; 
    float min_val; // TODO: 
    MEASURE_GPU_TIME(cudaMemcpy(&max_val, d_max_phase, sizeof(float), cudaMemcpyDeviceToHost), t_phase); 
    MEASURE_GPU_TIME(cudaMemcpy(&min_val, d_min_phase, sizeof(float), cudaMemcpyDeviceToHost), t_phase); 
    cout << "Max: " << max_val << endl; 
    cout << "Min: " << min_val << endl; 
    MEASURE_GPU_TIME((norm<<<grid_size, block_size>>>(d_image, d_buffer, max_val, min_val, N)), t_phase); 
    MEASURE_GPU_TIME(cudaMemcpy(idata, d_buffer, N * sizeof(uint8_t), cudaMemcpyDeviceToHost), t_oCopy); 

    // Merge channels and convert from HSV to BGR
    MEASURE_CPU_TIME(merge(hsv_channels, image), t_HSVtoBGR); 
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_HSV2BGR), t_HSVtoBGR);

    auto vevid_stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, milli> total = vevid_stop - vevid_start; 

    if (show_timing) {
        cout << "Timing Results: " << endl; 
        cout << "BGR to HSV time per frame: " << t_BGRtoHSV << " ms" << endl; 
        cout << "Forward data copy time per frame: " << t_iCopy << " ms" << endl; 
        cout << "FFT time per frame: " << t_fft << " ms" << endl; 
        cout << "Kernel multiplication time per frame: " << t_hadamard << " ms" << endl; 
        cout << "Inverse FFT time per frame: " << t_ifft << " ms" << endl; 
        cout << "Phase time per frame: " << t_phase << " ms" << endl; 
        cout << "Backward data copy time per frame: " << t_oCopy << " ms" << endl; 
        cout << "HSV to BGR time per frame: " << t_HSVtoBGR << " ms" << endl; 
        cout << "Total time per frame: " << total.count() << " ms" << endl; 
        cout << endl; 
        t_BGRtoHSV = 0; 
        t_iCopy = 0; 
        t_fft = 0; 
        t_hadamard = 0; 
        t_ifft = 0; 
        t_phase = 0; 
        t_oCopy = 0; 
        t_HSVtoBGR = 0; 
    }
}

Vevid::~Vevid()
{
    /*
    cudaFree(d_vevid_kernel); 
    cudaFree(d_image); 
    cudaFree(d_buffer); 
    cudaFree(d_max_phase); 
    cudaFree(d_min_phase); 
    */
}

struct Parameters { // TODO: change to b,G,S,T, explain meaning once
    float S; 
    float T; 
    float b;
    float G; 

    Parameters() : S(0), T(0), b(0), G(0)
    {}
};

struct Times {
    float t_BGRtoHSV; 
    float t_iCopy; 
    float t_fft; 
    float t_hadamard; 
    float t_ifft;
    float t_phase;  
    float t_oCopy; 
    float t_HSVtoBGR; 
}; 

struct VevidContext
{
    /* GPU resources */
    cufftComplex* d_vevid_kernel; 
    cufftComplex* d_image; 
    uint8_t* d_buffer; 
    cufftHandle m_plan; 
    float* d_max_phase; 
    float* d_min_phase; 

    /* Input parameters */
    Parameters params; 

    /* Frame width and height */
    size_t width; 
    size_t height; 
}; 

VevidContext context; 
Times times; 

void vevid_init(cv::Mat& image, int width, int height, float S, float T, float b, float G) 
{
    context.width = width; 
    context.height = height; 
    context.params.S = S; 
    context.params.T = T; 
    context.params.b = b; 
    context.params.G = G; 
    const int N = context.width * context.height; 

    cudaMalloc((void**)&context.d_vevid_kernel, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&context.d_image, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&context.d_buffer, N * sizeof(uint8_t)); 
    cufftPlan2d(&context.m_plan, height, width, CUFFT_C2C); 

    cudaMalloc((void**)&context.d_max_phase, sizeof(float)); 
    cudaMalloc((void**)&context.d_min_phase, sizeof(float)); 
    cudaMemset(context.d_max_phase, -10, sizeof(float)); // less than PI / 2
    cudaMemset(context.d_min_phase, 10, sizeof(float)); // greater than PI / 2

    int block_size = 32; 
    int grid_size = ((N) + block_size - 1) / block_size; 

    float* d_max; 
    float max_val;
    cudaMalloc((void**)&d_max, sizeof(float));  
    init_kernel<<<grid_size, block_size>>>(context.d_vevid_kernel, S, T, width, height); 
    max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_vevid_kernel, d_max, N);
    cudaMemcpy(&max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost);  
    scale_exp<<<grid_size, block_size>>>(context.d_vevid_kernel, (S / max_val), N);
    fftshift<<<grid_size, block_size>>>(context.d_vevid_kernel, width, height);
    cudaFree(d_max); 

    auto vevid_start = chrono::high_resolution_clock::now(); 
    
    // Convert from BGR to HSV and get V channel of image
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_BGR2HSV), times.t_BGRtoHSV);
    auto start = chrono::high_resolution_clock::now(); 
    vector<Mat> hsv_channels; 
    split(image, hsv_channels); 
    uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0); 
    auto stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, std::milli> elapsed = stop - start; 
    times.t_BGRtoHSV += elapsed.count(); 

    cudaMemset(context.d_buffer, 0, N * sizeof(uint8_t));
    MEASURE_GPU_TIME(cudaMemcpy(context.d_buffer, idata, N * sizeof(uint8_t), cudaMemcpyHostToDevice), times.t_iCopy); 

    
    // Call CUDA kernels

    // Take FFT
    MEASURE_GPU_TIME((populate<<<grid_size, block_size>>>(context.d_image, context.d_buffer, context.params.b, N)), times.t_fft);
    MEASURE_GPU_TIME(cufftExecC2C(context.m_plan, context.d_image, context.d_image, CUFFT_FORWARD), times.t_fft); 

    // Multiply kernel with image in frequency domain
    MEASURE_GPU_TIME((hadamard<<<grid_size, block_size>>>(context.d_vevid_kernel, context.d_image, N)), times.t_hadamard);

    // Take IFFT
    MEASURE_GPU_TIME(cufftExecC2C(context.m_plan, context.d_image, context.d_image, CUFFT_INVERSE), times.t_ifft);
    MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(context.d_image, (1.0f / N), N)), times.t_ifft);
    
    // Get phase
    MEASURE_GPU_TIME((phase<<<grid_size, block_size>>>(context.d_image, context.d_buffer, context.params.G, N)), times.t_phase);
    //cudaDeviceSynchronize();  
    MEASURE_GPU_TIME(cudaMemset(context.d_max_phase, -10, sizeof(float)), times.t_phase); // less than PI / 2
    MEASURE_GPU_TIME(cudaMemset(context.d_min_phase, 10, sizeof(float)), times.t_phase); // greater than PI / 2
    //cudaDeviceSynchronize(); 
    //MEASURE_GPU_TIME((min_max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_image, context.d_max_phase, context.d_min_phase, N)), times.t_phase);
    //cudaDeviceSynchronize();  
    //float max_phase; 
    //float min_phase; 
    //MEASURE_GPU_TIME(cudaMemcpy(&max_val, context.d_max_phase, sizeof(float), cudaMemcpyDeviceToHost), times.t_phase); 
    //MEASURE_GPU_TIME(cudaMemcpy(&min_val, context.d_min_phase, sizeof(float), cudaMemcpyDeviceToHost), times.t_phase); 
    //cout << "Max: " << max_val << endl; 
    //cout << "Min: " << min_val << endl; 
    MEASURE_GPU_TIME((norm<<<grid_size, block_size>>>(context.d_image, context.d_buffer, 0, -M_PI / 2.0f, N)), times.t_phase); 
    MEASURE_GPU_TIME(cudaMemcpy(idata, context.d_buffer, N * sizeof(uint8_t), cudaMemcpyDeviceToHost), times.t_oCopy); 

    // Merge channels and convert from HSV to BGR
    MEASURE_CPU_TIME(merge(hsv_channels, image), times.t_HSVtoBGR); 
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_HSV2BGR), times.t_HSVtoBGR);

    auto vevid_stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, milli> total = vevid_stop - vevid_start; 


    if (false) {
        cout << "Timing Results: " << endl; 
        cout << "BGR to HSV time per frame: " << times.t_BGRtoHSV << " ms" << endl; 
        cout << "Forward data copy time per frame: " << times.t_iCopy << " ms" << endl; 
        cout << "FFT time per frame: " << times.t_fft << " ms" << endl; 
        cout << "Kernel multiplication time per frame: " << times.t_hadamard << " ms" << endl; 
        cout << "Inverse FFT time per frame: " << times.t_ifft << " ms" << endl; 
        cout << "Phase time per frame: " << times.t_phase << " ms" << endl; 
        cout << "Backward data copy time per frame: " << times.t_oCopy << " ms" << endl; 
        cout << "HSV to BGR time per frame: " << times.t_HSVtoBGR << " ms" << endl; 
        cout << "Total time per frame: " << total.count() << " ms" << endl; 
        cout << endl; 
        times.t_BGRtoHSV = 0; 
        times.t_iCopy = 0; 
        times.t_fft = 0; 
        times.t_hadamard = 0; 
        times.t_ifft = 0; 
        times.t_phase = 0; 
        times.t_oCopy = 0; 
        times.t_HSVtoBGR = 0; 
    }

    cudaFree(context.d_vevid_kernel); 
    cudaFree(context.d_image); 
    cudaFree(context.d_buffer); 
    cufftDestroy(context.m_plan); 
    cudaFree(context.d_max_phase); 
    cudaFree(context.d_min_phase); 
}

void vevid(cv::Mat& image, bool show_timing, bool lite)
{
    cout << context.params.S << endl; 
    cout << context.params.T << endl; 
    cout << context.params.b << endl; 
    cout << context.params.G << endl; 

    auto vevid_start = chrono::high_resolution_clock::now(); 
    const int N = context.width * context.height; 
    
    // Convert from BGR to HSV and get V channel of image
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_BGR2HSV), times.t_BGRtoHSV);
    auto start = chrono::high_resolution_clock::now(); 
    vector<Mat> hsv_channels; 
    split(image, hsv_channels); 
    uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0); 
    auto stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, std::milli> elapsed = stop - start; 
    times.t_BGRtoHSV += elapsed.count(); 

    cudaMemset(context.d_buffer, 0, N * sizeof(uint8_t));
    MEASURE_GPU_TIME(cudaMemcpy(context.d_buffer, idata, N * sizeof(uint8_t), cudaMemcpyHostToDevice), times.t_iCopy); 

    
    // Call CUDA kernels
    int block_size = 32; 
    int grid_size = (N + block_size - 1) / block_size; 

    // Take FFT
    MEASURE_GPU_TIME((populate<<<grid_size, block_size>>>(context.d_image, context.d_buffer, context.params.b, N)), times.t_fft);
    MEASURE_GPU_TIME(cufftExecC2C(context.m_plan, context.d_image, context.d_image, CUFFT_FORWARD), times.t_fft); 

    // Multiply kernel with image in frequency domain
    MEASURE_GPU_TIME((hadamard<<<grid_size, block_size>>>(context.d_vevid_kernel, context.d_image, N)), times.t_hadamard);

    // Take IFFT
    MEASURE_GPU_TIME(cufftExecC2C(context.m_plan, context.d_image, context.d_image, CUFFT_INVERSE), times.t_ifft);
    MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(context.d_image, (1.0f / N), N)), times.t_ifft);
    
    // Get phase
    MEASURE_GPU_TIME((phase<<<grid_size, block_size>>>(context.d_image, context.d_buffer, context.params.G, N)), times.t_phase);
    //cudaDeviceSynchronize();  
    MEASURE_GPU_TIME(cudaMemset(context.d_max_phase, -10, sizeof(float)), times.t_phase); // less than PI / 2
    MEASURE_GPU_TIME(cudaMemset(context.d_min_phase, 10, sizeof(float)), times.t_phase); // greater than PI / 2
    //cudaDeviceSynchronize(); 
    MEASURE_GPU_TIME((min_max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_image, context.d_max_phase, context.d_min_phase, N)), times.t_phase);
    //cudaDeviceSynchronize();  
    float max_val; 
    float min_val; 
    MEASURE_GPU_TIME(cudaMemcpy(&max_val, context.d_max_phase, sizeof(float), cudaMemcpyDeviceToHost), times.t_phase); 
    MEASURE_GPU_TIME(cudaMemcpy(&min_val, context.d_min_phase, sizeof(float), cudaMemcpyDeviceToHost), times.t_phase); 
    //cout << "Max: " << max_val << endl; 
    //cout << "Min: " << min_val << endl; 
    MEASURE_GPU_TIME((norm<<<grid_size, block_size>>>(context.d_image, context.d_buffer, 0, -M_PI / 2.0f, N)), times.t_phase); 
    MEASURE_GPU_TIME(cudaMemcpy(idata, context.d_buffer, N * sizeof(uint8_t), cudaMemcpyDeviceToHost), times.t_oCopy); 

    // Merge channels and convert from HSV to BGR
    MEASURE_CPU_TIME(merge(hsv_channels, image), times.t_HSVtoBGR); 
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_HSV2BGR), times.t_HSVtoBGR);

    auto vevid_stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, milli> total = vevid_stop - vevid_start; 


    if (show_timing) {
        cout << "Timing Results: " << endl; 
        cout << "BGR to HSV time per frame: " << times.t_BGRtoHSV << " ms" << endl; 
        cout << "Forward data copy time per frame: " << times.t_iCopy << " ms" << endl; 
        cout << "FFT time per frame: " << times.t_fft << " ms" << endl; 
        cout << "Kernel multiplication time per frame: " << times.t_hadamard << " ms" << endl; 
        cout << "Inverse FFT time per frame: " << times.t_ifft << " ms" << endl; 
        cout << "Phase time per frame: " << times.t_phase << " ms" << endl; 
        cout << "Backward data copy time per frame: " << times.t_oCopy << " ms" << endl; 
        cout << "HSV to BGR time per frame: " << times.t_HSVtoBGR << " ms" << endl; 
        cout << "Total time per frame: " << total.count() << " ms" << endl; 
        cout << endl; 
        times.t_BGRtoHSV = 0; 
        times.t_iCopy = 0; 
        times.t_fft = 0; 
        times.t_hadamard = 0; 
        times.t_ifft = 0; 
        times.t_phase = 0; 
        times.t_oCopy = 0; 
        times.t_HSVtoBGR = 0; 
    }
}




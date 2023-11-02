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

// TODO: look into how timing scales for object detection with different resolutions.

Vevid::Vevid(int width, int height, float S, float T, float b, float G)
: m_width(width), m_height(height), m_S(S), m_T(T), m_b(b), m_G(G), d_vevid_kernel(nullptr), d_image(nullptr), d_buffer(nullptr), d_max_phase(nullptr), d_min_phase(nullptr),
t_BGRtoHSV(0), t_iCopy(0), t_fft(0), t_hadamard(0), t_ifft(0), t_phase(0), t_oCopy(0), t_HSVtoBGR(0)
{
    const int N = m_width * m_height; 

    cudaMalloc((void**)&d_vevid_kernel, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_image, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_buffer, N * sizeof(uint8_t)); 
    cufftPlan2d(&m_plan, m_height, m_width, CUFFT_C2C); 

    cudaMallocManaged((void**)&d_max_phase, sizeof(float));
    cudaMallocManaged((void**)&d_min_phase, sizeof(float));
    cudaMemset(d_max_phase, -10, sizeof(float)); // less than PI / 2
    cudaMemset(d_min_phase, 10, sizeof(float)); // greater than PI / 2

    int block_size = 32; 
    int grid_size = ((N) + block_size - 1) / block_size; 

    float* d_max;
    cudaMallocManaged((void**)&d_max, sizeof(float));  
    init_kernel<<<grid_size, block_size>>>(d_vevid_kernel, S, T, m_width, m_height);
    cudaDeviceSynchronize(); 
    max_reduce<<<64, block_size, block_size * sizeof(float)>>>(d_vevid_kernel, d_max, N);
    cudaDeviceSynchronize();
    scale_exp<<<grid_size, block_size>>>(d_vevid_kernel, (S / *d_max), N);
    fftshift<<<grid_size, block_size>>>(d_vevid_kernel, m_width, m_height);
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

    if (lite) {
        // Get phase approximation
        MEASURE_GPU_TIME((populate<<<grid_size, block_size>>>(d_image, d_buffer, m_b, N)), t_phase);
        MEASURE_GPU_TIME((vevid_phase<<<grid_size, block_size>>>(d_image, d_buffer, m_G, N)), t_phase);
    }
    else {
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
    }

    MEASURE_GPU_TIME(cudaMemset(d_max_phase, -10, sizeof(float)), t_phase); // less than PI / 2
    MEASURE_GPU_TIME(cudaMemset(d_min_phase, 10, sizeof(float)), t_phase); // greater than PI / 2
    cudaDeviceSynchronize(); 
    MEASURE_GPU_TIME((min_max_reduce<<<64, block_size, block_size * sizeof(float)>>>(d_image, d_max_phase, d_min_phase, N)), t_phase);
    cudaDeviceSynchronize();
    MEASURE_GPU_TIME((norm<<<grid_size, block_size>>>(d_image, d_buffer, *d_max_phase, *d_min_phase, N)), t_phase); 
    //MEASURE_GPU_TIME((norm<<<grid_size, block_size>>>(d_image, d_buffer, 0, -M_PI / 2, N)), t_phase); 
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
    cudaFree(d_vevid_kernel); 
    cudaFree(d_image); 
    cudaFree(d_buffer); 
}



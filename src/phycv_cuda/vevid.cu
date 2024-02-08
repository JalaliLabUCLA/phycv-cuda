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

Vevid::Vevid(int width, int height, float S, float T, float b, float G)
: m_width(width), m_height(height), m_S(S), m_T(T), m_b(b), m_G(G), d_data(nullptr), d_vevid_kernel(nullptr), d_image_V(nullptr), d_max_phase(nullptr), d_min_phase(nullptr),
t_BGRtoHSV(0), t_iCopy(0), t_fft(0), t_hadamard(0), t_ifft(0), t_phase(0), t_oCopy(0), t_HSVtoBGR(0)
{
    const int N = m_width * m_height; 

    cudaMalloc((void**)&d_data, N * 3 * sizeof(uint8_t));
    cudaMalloc((void**)&d_vevid_kernel, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_image_V, N * sizeof(cufftComplex)); 
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

void Vevid::run(Mat& image, bool show_timing, bool lite)
{
    uint8_t* idata =  image.ptr<uint8_t>(0);
    auto vevid_start = chrono::high_resolution_clock::now(); 
    const int N = m_width * m_height; 

    dim3 conversion_block_size(16, 16); 
    dim3 conversion_grid_size((image.cols + conversion_block_size.x - 1) / conversion_block_size.x, (image.rows + conversion_block_size.y - 1) / conversion_block_size.y); 
    int block_size = 32; 
    int grid_size = (N + block_size - 1) / block_size; 
    
    // Copy data to device
    MEASURE_GPU_TIME(cudaMemcpy(d_data, image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice), t_iCopy);

    // Convert from BGR to HSV
    MEASURE_GPU_TIME((BGR2HSVKernel<<<conversion_grid_size, conversion_block_size>>>(d_data, image.cols, image.rows, image.step)), t_BGRtoHSV);

    if (lite) {
        // Get phase approximation
        MEASURE_GPU_TIME((populate<<<grid_size, block_size>>>(d_image_V, d_data, m_b, N)), t_phase);
        MEASURE_GPU_TIME((vevid_phase<<<grid_size, block_size>>>(d_image_V, d_data, m_G, N)), t_phase);
    }
    else {
        // Take FFT
        MEASURE_GPU_TIME((populate<<<grid_size, block_size>>>(d_image_V, d_data, m_b, N)), t_fft);
        MEASURE_GPU_TIME(cufftExecC2C(m_plan, d_image_V, d_image_V, CUFFT_FORWARD), t_fft); 

        // Multiply kernel with image in frequency domain
        MEASURE_GPU_TIME((hadamard<<<grid_size, block_size>>>(d_vevid_kernel, d_image_V, N)), t_hadamard);

        // Take IFFT
        MEASURE_GPU_TIME(cufftExecC2C(m_plan, d_image_V, d_image_V, CUFFT_INVERSE), t_ifft);
        MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(d_image_V, (1.0f / (N)), N)), t_ifft);
        
        // Get phase
        MEASURE_GPU_TIME((phase<<<grid_size, block_size>>>(d_image_V, d_data, m_G, N)), t_phase);
    }

    // Normalize V-channel values
    MEASURE_GPU_TIME((min_max_reduce<<<64, block_size, block_size * sizeof(float)>>>(d_image_V, d_max_phase, d_min_phase, N)), t_phase);
    cudaDeviceSynchronize();
    MEASURE_GPU_TIME((norm<<<grid_size, block_size>>>(d_image_V, d_data, *d_max_phase, *d_min_phase, N)), t_phase);  

    // Convert from HSV to BGR
    MEASURE_GPU_TIME((HSV2BGRKernel<<<conversion_grid_size, conversion_block_size>>>(d_data, image.cols, image.rows, image.step)), t_HSVtoBGR); 

    // Copy data to host
    MEASURE_GPU_TIME(cudaMemcpy(idata, d_data, N * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost), t_oCopy); 
    cudaDeviceSynchronize();
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
    cudaFree(d_image_V); 
    cudaFree(d_data); 
}
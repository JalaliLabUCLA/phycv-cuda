#ifndef VEVID_CUH
#define VEVID_CUH

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

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


class Vevid {
public: 
    Vevid(int width, int height, float S, float T, float b, float G); 
    ~Vevid(); 
    void run(cv::Mat& image, bool show_timing, bool lite); 

private: 
    int m_width; 
    int m_height; 

    // Input parameters
    float m_S; 
    float m_T; 
    float m_b; 
    float m_G; 

    // GPU resources
    uint8_t* d_data; 
    cufftComplex* d_vevid_kernel; 
    cufftComplex* d_image_V; 
    cufftHandle m_plan; 
    float* d_max_phase; 
    float* d_min_phase; 

    // Timing values
    float t_BGRtoHSV; 
    float t_iCopy; 
    float t_fft; 
    float t_hadamard; 
    float t_ifft;
    float t_phase;  
    float t_oCopy; 
    float t_HSVtoBGR; 
};

#endif // VEVID_CUH
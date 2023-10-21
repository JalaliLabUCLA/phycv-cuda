#ifndef VEVID_CUH
#define VEVID_CUH

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

void vevid_init(int width, int height, float phase_strength, float spectral_phase_variance, float regularization_term, float phase_activation_gain);
void vevid(cv::Mat& image, bool show_timing, bool lite);
void vevid_fini();


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
    cufftComplex* d_vevid_kernel; 
    cufftComplex* d_image; 
    uint8_t* d_buffer; 
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

#endif // VEVID_H

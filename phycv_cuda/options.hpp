#ifndef OPTIONS_H
#define OPTIONS_H

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct Flags {
    char* pvalue = nullptr; 
    char* ivalue = nullptr; 
    char* vvalue = nullptr; 
    char* wvalue = nullptr; 
    bool lflag = false; 
    bool dflag = false; 
    bool tflag = false; 
    bool hflag = false; 
};

struct Params {
    int width = 640; 
    int height = 480; 
    float S = 0.8; // S < 1 (0.5 - 1), smaller S, stronger enhancement, decreasing increases noise
    float T = 0.01; // t should be between 0.01 and 0.001
    float b = 0.2; // b should be between 0.1 and 0.4, increasing suppresses noise
    float G = 1; // (inverse gain), smaller G, stronger enhancement, decreasing increases noise
}; 

void print_usage(const char* program_name);
void process_args(int argc, char* argv[], Flags* flags, Params* params);
void process_image(cv::Mat& frame, Flags* flags, Params* params, bool show_detections); // move to video.cpp
void process_video(cv::VideoCapture& camera, cv::Mat& frame, Flags* flags, Params* params, bool show_detections); // move to video.cpp 

#endif // OPTIONS_H
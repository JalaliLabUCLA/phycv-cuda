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
    float S = 10; 
    float T = 0.1; 
    float b = 4; 
    float G = 10; 
}; 

void print_usage(const char* program_name);
void process_args(int argc, char* argv[], Flags* flags, Params* params);
void process_image(cv::Mat& frame, Flags* flags, Params* params, bool show_detections); 
void process_video(cv::VideoCapture& camera, cv::Mat& frame, Flags* flags, Params* params, bool show_detections); 

#endif // OPTIONS_H
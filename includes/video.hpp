#ifndef VIDEO_HPP
#define VIDEO_HPP

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <mutex>
#include <condition_variable>

#include "options.hpp"

class WebCam {
public: 
    WebCam(int device, int width, int height); 
    ~WebCam(); 
    void start_capturing(); 
    void stop_capturing(); 
    cv::Mat get_frame(); 
    int get_frame_count() const; 
    int get_width() const; 
    int get_height() const; 
    cv::VideoCapture m_webcam; 

private: 
    
    int m_device; 
    int m_width; 
    int m_height; 
    int m_frame_count; 
    bool m_exit; 
    bool m_new_frame; 
    cv::Mat m_frame; 
    std::mutex m_mutex; 
    std::condition_variable m_frame_cv;
}; 

class Window {
public:
    Window(const std::string &window1_name, const std::string& window2_name); 
    void process_camera(WebCam &webcam, Flags* flags, Params* params); 
    void detect_objects(cv::Mat image, int width, int height);
    void process_image(cv::Mat& frame, Flags* flags, Params* params); 
    void process_video(cv::VideoCapture& camera, cv::Mat& frame, Flags* flags, Params* params);


private: 
    void display_fps(cv::Mat& frame);
    std::string m_window1_name;
    std::string m_window2_name;
    bool m_exit;  
    int m_frame_count; 
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_end_time;
    double m_total_frame_time;
}; 

#endif // VIDEO_HPP
#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <mutex>
#include <condition_variable>

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

private: 


    cv::VideoCapture m_webcam; 
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
    void start_display(WebCam &webcam, bool show_fps, bool show_detections, bool show_timing, bool lite); 
    void detect_objects(cv::Mat image, int width, int height);

private: 
    void display_fps(cv::Mat& frame);
    std::string m_window1_name;
    std::string m_window2_name;
    bool m_exit; 
    bool m_show_fps; 
    int m_frame_count; 
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_end_time;
    double m_total_frame_time;
}; 

#endif // VIDEO_H

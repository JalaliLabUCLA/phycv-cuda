#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <iomanip>

#include "vevid.cuh"
#include "video.hpp"

using namespace std; 
using namespace cv; 

WebCam::WebCam(int device, int width, int height)
    : m_device(device), m_width(width), m_height(height), m_frame_count(0), m_exit(false), m_new_frame(false) 
{
        m_webcam.open(m_device, CAP_V4L2);

        m_webcam.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
        m_webcam.set(CAP_PROP_FRAME_WIDTH, m_width);
        m_webcam.set(CAP_PROP_FRAME_HEIGHT, m_height);
        m_webcam.set(CAP_PROP_FPS, 30); 
}

WebCam::~WebCam() {
    m_webcam.release();
}

void WebCam::start_capturing() {
    while (!m_exit) {
        Mat frame; 
        m_webcam >> frame;
        unique_lock<mutex> lock(m_mutex);
        m_new_frame = true;
        m_frame = frame; 
        lock.unlock();
        m_frame_cv.notify_one();
        m_frame_count++;
    }
}

void WebCam::stop_capturing() {
    m_exit = true;
}

Mat WebCam::get_frame() {
    unique_lock<mutex> lock(m_mutex);
    m_frame_cv.wait(lock, [this]() { return m_new_frame; });
    m_new_frame = false;
    return m_frame;
}

int WebCam::get_frame_count() const {
    return m_frame_count;
}

int WebCam::get_width() const {
    return m_width; 
}

int WebCam::get_height() const {
    return m_height; 
}

Window::Window(const string& window1_name, const string& window2_name)
    : m_window1_name(window1_name), m_window2_name(window2_name), m_exit(false), m_show_fps(false), m_frame_count(0) 
{
    namedWindow(m_window1_name, WINDOW_NORMAL); 
    namedWindow(m_window2_name, WINDOW_NORMAL);
}

void Window::start_display(WebCam& webcam, bool show_fps) {

    vevid_init(webcam.get_width(), webcam.get_height(), 10, 0.1, 4, 2.2); 

    m_start_time = std::chrono::steady_clock::now();

    while (!m_exit) {
        Mat frame = webcam.get_frame();

        if (show_fps) {
            display_fps(frame); 
        } 

        imshow(m_window1_name, frame); 

        vevid(frame); 

        imshow(m_window2_name, frame); 
        m_frame_count++;

        char key = waitKey(1);
        if (key == 27) {
            m_exit = true;
        }
    }

    m_end_time = std::chrono::steady_clock::now();  
    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(m_end_time - m_start_time).count();
    double average_frame_time = elapsed_time / m_frame_count;

    std::cout << "Average Time Per Frame: " << average_frame_time << " ms" << std::endl;

    vevid_fini(); 
}

void Window::display_fps(cv::Mat& frame) {
    static chrono::steady_clock::time_point last_time = chrono::steady_clock::now();
    chrono::steady_clock::time_point current_time = chrono::steady_clock::now();
    chrono::duration<double> duration = current_time - last_time;
    double fps = 1.0 / duration.count();

    ostringstream ss;
    ss << "FPS: " << fixed << setprecision(1) << fps;

    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 3;
    int thickness = 4;
    int baseline = 0;
    Size text_size = getTextSize(ss.str(), font_face, font_scale, thickness, &baseline);

    Point text_org(frame.cols - text_size.width - 10, frame.rows - text_size.height - 10);
    putText(frame, ss.str(), text_org, font_face, font_scale, Scalar(255, 255, 255), thickness);

    last_time = current_time;
}

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudaRGB.h>
#include <cufft.h>
#include <detectNet.h>
#include <objectTracker.h>
#include <signal.h>

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
        m_webcam.open(m_device, CAP_ANY);

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

    detectNet* net = detectNet::Create("ssd-mobilenet-v2", 0.5); 
    uchar3* d_image; 
    cudaMalloc((void**)&d_image, webcam.get_width() * webcam.get_height() * sizeof(uchar3));

    m_start_time = std::chrono::steady_clock::now();

    while (!m_exit) {
        Mat frame = webcam.get_frame();

        if (show_fps) {
            display_fps(frame); 
        }

        imshow(m_window1_name, frame);

        cudaDeviceSynchronize(); 
        vevid(frame); 
        cudaDeviceSynchronize(); 

        
        cvtColor(frame, frame, COLOR_BGR2RGB); 
        cudaDeviceSynchronize(); 
        cudaMemcpy2D(d_image, webcam.get_width()  * sizeof(uchar3), frame.data, frame.step, webcam.get_width() * sizeof(uchar3), webcam.get_height(), cudaMemcpyHostToDevice);
        detectNet::Detection* detections = NULL; 
        const int numDetections = net->Detect(d_image, webcam.get_width(), webcam.get_height(), &detections);
        cudaDeviceSynchronize(); 
        cvtColor(frame, frame, COLOR_RGB2BGR); 
        
        if( numDetections > 0 )
		{
			LogVerbose("%i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
				LogVerbose("\ndetected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				LogVerbose("bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
			
				if( detections[n].TrackID >= 0 ) // is this a tracked object?
					LogVerbose("tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections[n].TrackID, detections[n].TrackStatus, detections[n].TrackFrames, detections[n].TrackLost);
			}
		}

        imshow(m_window2_name, frame); 
        m_frame_count++;

        char key = waitKey(1);
        if (key == 27) {
            m_exit = true;
        }
    }

    m_end_time = chrono::steady_clock::now();  
    double elapsed_time = chrono::duration_cast<chrono::milliseconds>(m_end_time - m_start_time).count();
    double average_frame_time = elapsed_time / m_frame_count;

    std::cout << "Average Time Per Frame: " << average_frame_time << " ms" << std::endl;

    vevid_fini(); 
}


void Window::display_fps(Mat& frame) {
    static chrono::steady_clock::time_point last_time = chrono::steady_clock::now();
    chrono::steady_clock::time_point current_time = chrono::steady_clock::now();
    chrono::duration<double> duration = current_time - last_time;
    double fps = 1.0 / duration.count();

    ostringstream ss;
    ss << "FPS: " << fixed << setprecision(1) << fps;

    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 2;
    int thickness = 4;
    int baseline = 0;
    Size text_size = getTextSize(ss.str(), font_face, font_scale, thickness, &baseline);

    Point text_org(frame.cols - text_size.width - 10, frame.rows - text_size.height - 10);
    putText(frame, ss.str(), text_org, font_face, font_scale, Scalar(255, 255, 255), thickness);

    last_time = current_time;
}
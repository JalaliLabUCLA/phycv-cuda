#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <getopt.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <detectNet.h>
#include <objectTracker.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudaRGB.h>

#include "vevid.cuh"
#include "video.hpp"
#include "options.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    /*
    Vevid vevid(4, 4, 1, 1, 1, 1);

    cv::Mat values = cv::Mat(1, 16, CV_8UC1);
    for (int i = 0; i < 16; i++) {
        values.at<uchar>(0, i) = static_cast<uchar>(i + 1);
    }
    cv::Mat array_3channel = cv::Mat::zeros(4, 4, CV_8UC3);

    // Set the same values in all three channels
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                array_3channel.at<cv::Vec3b>(i, j)[c] = values.at<uchar>(0, i * 4 + j);
            }
        }
    }

    for (int i = 0; i < array_3channel.rows; i++) {
    for (int j = 0; j < array_3channel.cols; j++) {
        cv::Vec3b pixel = array_3channel.at<cv::Vec3b>(i, j);
        uchar blue = pixel[0];
        uchar green = pixel[1];
        uchar red = pixel[2];
        std::cout << "Pixel (" << i << ", " << j << "): B=" << static_cast<int>(blue)
                  << " G=" << static_cast<int>(green) << " R=" << static_cast<int>(red) << std::endl;
        }
    }

    cout << endl; 
    cout << "===" << endl; 
    vevid.run(array_3channel, false, false); 


    */
    Flags flags; 
    Params params; 
    process_args(argc, argv, &flags, &params); 

    vevid_init(params.width, params.height, params.S, params.T, params.b, params.G); 
    Mat frame; 

    if (flags.ivalue != nullptr) {
        process_image(frame, &flags, &params, flags.dflag); 
        vevid_fini(); 
        return 0; 
    }

    VideoCapture camera;

    if (flags.vvalue != nullptr) {
        process_video(camera, frame, &flags, &params, flags.dflag); 
        vevid_fini(); 
        return 0; 
    }
    
    WebCam webcam(0, params.width, params.height); 
    Window window("Original Video", "Vevid-Enhanced Video"); 
    thread capture_thread(&WebCam::start_capturing, &webcam); 
    thread display_thread(&Window::start_display, &window, ref(webcam), true, flags.dflag, flags.tflag, flags.lflag);
    display_thread.join(); 
    webcam.stop_capturing();
    capture_thread.join(); 

    destroyAllWindows(); 
    
    return 0; 
}
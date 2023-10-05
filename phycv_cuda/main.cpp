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
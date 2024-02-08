#include <thread>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "vevid.cuh"
#include "video.hpp"
#include "options.hpp"
#include "detect_net.hpp"

using namespace std; 
using namespace cv; 

int main(int argc, char** argv) {
    
    Flags flags;
    Params params;

    process_args(argc, argv, &flags, &params); 

    WebCam webcam(0, params.width, params.height); 
    Window window("Original Video", "VEViD-Enhanced Video");

    // Process a single image
    Mat frame; 

    if (flags.i_value != nullptr) {
        window.process_image(frame, &flags, &params);
        return 0; 
    }

    // Process a single video
    VideoCapture camera;

    if (flags.v_value != nullptr) {
        window.process_video(camera, frame, &flags, &params); 
        return 0; 
    }
    
    // Process a camera feed
    thread capture_thread(&WebCam::start_capturing, &webcam); 
    thread display_thread(&Window::process_camera, &window, ref(webcam), &flags, &params);
    display_thread.join(); 
    webcam.stop_capturing();
    capture_thread.join(); 

    destroyAllWindows(); 
    return 0; 
}
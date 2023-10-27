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

// TODO: Resolve OO Vevid issues -- if not possible, revert to old version and solve fft scaling issue. 

int main(int argc, char** argv) {
    
    Flags flags; 
    Params params; 

    process_args(argc, argv, &flags, &params); 

    WebCam webcam(0, params.width, params.height); 
    cout << "Width: " << params.width << ", Height: " << params.height << endl; 
    Window window("Original Video", "VEViD-Enhanced Video");

    Mat frame; 

    if (flags.i_value != nullptr) {
        window.process_image(frame, &flags, &params, flags.d_flag); 
        return 0; 
    }

    VideoCapture camera;

    if (flags.v_value != nullptr) {
        window.process_video(camera, frame, &flags, &params, flags.d_flag); 
        return 0; 
    }
    
    thread capture_thread(&WebCam::start_capturing, &webcam); 
    thread display_thread(&Window::start_display, &window, ref(webcam), &params, true, flags.d_flag, flags.t_flag, flags.l_flag);
    display_thread.join(); 
    webcam.stop_capturing();
    capture_thread.join(); 

    destroyAllWindows(); 
    return 0; 
    
}
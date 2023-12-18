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

// TODO: 
// Make sure to push only PhyCV_CUDA parts of project to remote repo
// Add -r (xres, yres) flag
// Allow -p to specify individual parameters (-p S=val should change only S to val)
// Tracking -- implement simple logic (check if center is left/right/above/below bounding box) for pan/tilt to center of bounding box

int main(int argc, char** argv) {
    
    Flags flags;
    Params params;

    process_args(argc, argv, &flags, &params); 

    WebCam webcam(0, params.width, params.height); 
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
    thread display_thread(&Window::process_camera, &window, ref(webcam), &params, true, flags.d_flag, flags.t_flag, flags.l_flag);
    display_thread.join(); 
    webcam.stop_capturing();
    capture_thread.join(); 

    destroyAllWindows(); 
    return 0; 
    
}


/*
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>

#include "controls.hpp"

#define CHIP_I2C_ADDR 0x0C
#define BUS "/dev/i2c-1"

using namespace std; 
using namespace cv; 

int main(int argc, char** argv) {
    
    namedWindow("test", WINDOW_NORMAL); 

    Controller controller; 

    int motor_step = 5; 
    int focus_step = 100; 
    int zoom_step = 100; 

    while (true) {
        char key = waitKey(1); 
        cout << key << endl; 
        if (key == 's') {
            controller.set(OPT_MOTOR_Y, controller.get(OPT_MOTOR_Y) + motor_step); 
        }
        else if (key == 'w') {
            controller.set(OPT_MOTOR_Y, controller.get(OPT_MOTOR_Y) - motor_step); 
        }
        else if (key == 'd') {
            controller.set(OPT_MOTOR_X, controller.get(OPT_MOTOR_X) - motor_step); 
        }
        else if (key == 'a') {
            controller.set(OPT_MOTOR_X, controller.get(OPT_MOTOR_X) + motor_step);
        }
        else if (key == 'r') {
            controller.reset(OPT_FOCUS); 
            controller.reset(OPT_ZOOM); 
        }
        else if (key == 'R') {
            controller.set(OPT_ZOOM, controller.get(OPT_ZOOM) + zoom_step); 
        }
        else if (key == 'T') {
            controller.set(OPT_ZOOM, controller.get(OPT_ZOOM) - zoom_step); 
        }
        else if (key == 'Q') {
            controller.set(OPT_FOCUS, controller.get(OPT_FOCUS) - focus_step); 
        }
        else if (key == 'S') {
            controller.set(OPT_FOCUS, controller.get(OPT_FOCUS) + focus_step); 
        }
        else if (key == 27){
            break; 
        }
    }
    
    return 0;
}
*/
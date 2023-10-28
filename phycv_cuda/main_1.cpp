#include <thread>
#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "vevid.cuh"
#include "options.hpp"
#include "detect_net.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    
    Flags flags; 
    Params params; 

    process_args(argc, argv, &flags, &params); 

    Vevid vevid(params.width, params.height, params.S, params.T, params.b, params.G);


    namedWindow("Original", WINDOW_NORMAL); 
    namedWindow("VEViD-Enhanced", WINDOW_NORMAL); 

    if (flags.i_value != nullptr) {
        cout << flags.i_value << endl; 
        Mat frame = imread(flags.i_value); 
        resize(frame, frame, Size(params.width, params.height)); 
        imshow("Original", frame); 
        vevid.run(frame, false, false); 
        imshow("VEViD-Enhanced", frame); 
        waitKey(); 

        frame = imread(flags.i_value); 
        resize(frame, frame, Size(params.width, params.height)); 
        imshow("Original", frame); 
        Mat output_frame = frame.clone(); 
        vevid.run(output_frame, false, false); 
        imshow("VEViD-Enhanced", output_frame); 
        waitKey(); 
        return 0; 
    }
    
    if (flags.v_value != nullptr) {
        return 0; 
    }
    return 0; 
}
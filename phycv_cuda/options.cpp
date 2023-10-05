#include <iostream>
#include <getopt.h>
#include <detectNet.h>
#include <objectTracker.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudaRGB.h>

#include "options.hpp"
#include "vevid.cuh"

using namespace std; 
using namespace cv; 

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "   -p [ARGS]  : Specify input parameters (separate ARGS with commas)" << endl; 
    cout << "       ARGS: <W>,<H>,<S>,<T>,<b>,<G>" << endl; 
    cout << "           W (int):    Image width" << endl; 
    cout << "           H (int):    Image height" << endl; 
    cout << "           S (float):  Phase strength" << endl; 
    cout << "           T (float):  Variance of the spectral phase function" << endl; 
    cout << "           b (float):  Regularization term" << endl; 
    cout << "           G (float):  Phase activation gain" << endl; 
    cout << endl; 
    cout << "   -i <FILE>  : Run VEViD on an input image FILE" << endl;
    cout << "   -v <FILE>  : Run VEViD on on input video FILE" << endl;
    cout << "   -w <FILE>  : Write output to FILE" << endl; 
    cout << "   -l         : Run VEViD-lite" << endl;
    cout << "   -d         : Run object detection network" << endl;
    cout << "   -t         : Display timing info" << endl;
    cout << "   -h         : Print this help message" << endl;
    cout << endl; 
    cout << "   -i and -v options are mutually exclusive. Use only one of them at a time." << endl;
}

void process_args(int argc, char* argv[], Flags* flags, Params* params) {
    int c;
    bool specified = false; 
    while ((c = getopt(argc, argv, ":p:i:v:w:ldth")) != -1) {
        specified = true; 
        switch (c) {
            case 'p': 
                flags->pvalue = optarg; 
                break; 
            case 'i':
                flags->ivalue = optarg;
                break;
            case 'v':
                flags->vvalue = optarg;
                break;
            case 'w': 
                flags->wvalue = optarg; 
                break; 
            case 'l':
                flags->lflag = true;
                break;
            case 'd':
                flags->dflag = true;
                break;
            case 't':
                flags->tflag = true; 
                break;
            case 'h':
                flags->hflag = true;  
                break;
            case ':':
                cout << "option -" << (char)optopt << " requires an argument" << endl; 
                print_usage(argv[0]); 
                exit(0); 
            case '?':
                cout << "option -" << (char)optopt << " is not a valid option" << endl; 
                print_usage(argv[0]); 
                exit(0); 
            default:
                abort();
        }
    }

    if (flags->hflag == true) {
        print_usage(argv[0]); 
        exit(0); 
    }

    if (flags->tflag == true) {
        cout << "Timing information will be displayed" << endl; 
    }

    if (flags->ivalue != nullptr && flags->vvalue != nullptr) {
        cout << "both -i and -v flags specified" << endl; 
        print_usage(argv[0]); 
        exit(0); 
    }

    if (flags->pvalue == nullptr) {
        cout << "Custom parameters not specified, using default values:" << endl; 
        cout << "   width = " << params->width << endl; 
        cout << "   height = " << params->height << endl; 
        cout << "   S = " << params->S << endl; 
        cout << "   T = " << params->T << endl; 
        cout << "   b = " << params->b <<endl; 
        cout << "   G = " << params->G << endl; 
    }
    else {
        string input(flags->pvalue); 
        istringstream iss(input); 
        string token; 

        int int_count = 0; 
        int float_count = 0; 

        while(getline(iss, token, ',')) {
            if (int_count < 2) {
                int int_value; 
                try {
                    int_value = stoi(token); 
                }
                catch (const invalid_argument& e) {
                    cout << "Invalid integer value in custom parameters: " << token << endl; 
                    print_usage(argv[0]); 
                    exit(0); 
                }
                switch (int_count) {
                    case 0: 
                        params->width = int_value; 
                        break; 
                    case 1: 
                        params->height = int_value; 
                        break; 
                }
                int_count++; 
            }
            else if (float_count < 4) {
                float float_value; 
                try {
                    float_value = stof(token); 
                }
                catch (const invalid_argument& e) {
                    cout << "Invalid float value in custom parameters: " << token << endl; 
                    print_usage(argv[0]); 
                    exit(0); 
                }
                switch (float_count) {
                    case 0: 
                        params->S = float_value; 
                        break; 
                    case 1: 
                        params->T = float_value; 
                        break; 
                    case 2: 
                        params->b = float_value; 
                        break; 
                    case 3:
                        params->G = float_value; 
                        break; 
                }
                float_count++; 
            }
            else {
                cout << "Too many custom parameters" << endl; 
                print_usage(argv[0]); 
                exit(0); 
            }
        }

        if (int_count != 2 || float_count != 4) {
            cout << "Too few custom parameters" << endl; 
            print_usage(argv[0]); 
            exit(0); 
        }

        cout << "Custom parameters specified, using:" << endl;
        cout << "   width = " << params->width << endl;
        cout << "   height = " << params->height << endl;
        cout << "   S = " << params->S << endl;
        cout << "   T = " << params->T << endl;
        cout << "   b = " << params->b << endl;
        cout << "   G = " << params->G << endl;
    }
}

void process_image(Mat& frame, Flags* flags, Params* params, bool show_detections) {
    cout << "Running VEViD on input image " << flags->ivalue << endl; 

    namedWindow("Original Image", WINDOW_NORMAL); 
    namedWindow("VEViD-Enhanced Image", WINDOW_NORMAL); 

    frame = imread(flags->ivalue); 

    if (frame.empty()) {
        cout << "Error: Could not load the input image " << endl; 
        exit(1); 
    }

    uchar3* d_image; 
    detectNet* net; 
    if (show_detections) {
        net = detectNet::Create("ssd-mobilenet-v2", 0.5); 
        cudaMalloc((void**)&d_image, params->width * params->height * sizeof(uchar3));
    }

    resize(frame, frame, Size(params->width, params->height)); 
    imshow("Original Image", frame); 
    vevid(frame, false, flags->lflag); 

    if (show_detections) {
        cvtColor(frame, frame, COLOR_BGR2RGB); 
        cudaDeviceSynchronize(); 
        cudaMemcpy2D(d_image, params->width * sizeof(uchar3), frame.data, frame.step, params->width * sizeof(uchar3), params->height, cudaMemcpyHostToDevice);
        detectNet::Detection* detections = NULL; 
        const int numDetections = net->Detect(d_image, params->width, params->height, &detections);
        cudaDeviceSynchronize(); 
        cvtColor(frame, frame, COLOR_RGB2BGR); 
        
        std::chrono::steady_clock::time_point vevid_end = std::chrono::steady_clock::now(); 

        if( numDetections > 0 )
        {
            LogVerbose("%i objects detected\n", numDetections);
        
            for( int n=0; n < numDetections; n++ )
            {
                LogVerbose("\ndetected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
                LogVerbose("bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
            
                if( detections[n].TrackID >= 0 ) // is this a tracked object?
                    LogVerbose("tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections[n].TrackID, detections[n].TrackStatus, detections[n].TrackFrames, detections[n].TrackLost);
            
                rectangle(frame, Point(detections[n].Left, detections[n].Top), Point(detections[n].Right, detections[n].Bottom), Scalar(0, 255, 0), 2);
            }
        }
    }

    imshow("VEViD-Enhanced Image", frame); 

    waitKey();

    if (flags->wvalue != nullptr) {
        cout << "Writing image to " << flags->wvalue << endl;

        if (!imwrite(flags->wvalue, frame)) {
            cout << "Error: Could not write the output image to " << flags->wvalue << endl; 
            exit(1); 
        } 
    }
}

void process_video(VideoCapture& camera, Mat& frame, Flags* flags, Params* params, bool show_detections) {
    cout << "Running VEViD on input video " << flags->vvalue << endl;

    camera.open(flags->vvalue); 
    if (!camera.isOpened()) {
        cout << "Error: Could not open video file" << flags->vvalue << endl; 
        exit(1); 
    }

    bool change_dims = false; 
    if (camera.get(CAP_PROP_FRAME_WIDTH) != params->width ||
        camera.get(CAP_PROP_FRAME_HEIGHT) != params->height) 
    {
        change_dims = true; 
        cout << resize << endl; 
    }

    namedWindow("Original Video", WINDOW_NORMAL); 
    namedWindow("VEViD-Enhanced Video", WINDOW_NORMAL);

    VideoWriter output; 

    if (flags->wvalue != nullptr) {
        string output_path = flags->wvalue; 
        int fourcc = VideoWriter::fourcc('a', 'v', 'c', '1'); 
        int fps = 30; 
        Size frame_size(params->width, params->height); 

        output.open(output_path, fourcc, fps, frame_size); 

        if (!output.isOpened()) {
            cout << "Error: Could not write video to " << flags->wvalue << endl; 
            exit(1); 
        }
    }

    uchar3* d_image; 
    detectNet* net; 
    if (show_detections) {
        net = detectNet::Create("ssd-mobilenet-v2", 0.5); 
        cudaMalloc((void**)&d_image, params->width * params->height * sizeof(uchar3));
    }

    while (true) {
        camera >> frame; 

        if (frame.empty()) {
            break; 
        }

        if (change_dims) {
            resize(frame, frame, Size(params->width, params->height)); 
        }

        imshow("Original Video", frame); 
        vevid(frame, false, flags->lflag); 

        if (show_detections) {
            cvtColor(frame, frame, COLOR_BGR2RGB); 
            cudaDeviceSynchronize(); 
            cudaMemcpy2D(d_image, params->width * sizeof(uchar3), frame.data, frame.step, params->width * sizeof(uchar3), params->height, cudaMemcpyHostToDevice);
            detectNet::Detection* detections = NULL; 
            const int numDetections = net->Detect(d_image, params->width, params->height, &detections);
            cudaDeviceSynchronize(); 
            cvtColor(frame, frame, COLOR_RGB2BGR); 
            
            std::chrono::steady_clock::time_point vevid_end = std::chrono::steady_clock::now(); 

            if( numDetections > 0 )
            {
                LogVerbose("%i objects detected\n", numDetections);
            
                for( int n=0; n < numDetections; n++ )
                {
                    LogVerbose("\ndetected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
                    LogVerbose("bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
                
                    if( detections[n].TrackID >= 0 ) // is this a tracked object?
                        LogVerbose("tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections[n].TrackID, detections[n].TrackStatus, detections[n].TrackFrames, detections[n].TrackLost);
                
                    rectangle(frame, Point(detections[n].Left, detections[n].Top), Point(detections[n].Right, detections[n].Bottom), Scalar(0, 255, 0), 2);
                }
            }
        }

        imshow("VEViD-Enhanced Video", frame); 
        
        if (flags->wvalue != nullptr) {
            output.write(frame); 
        }

        char key = waitKey(1);
        if (key == 27) {
            break; 
        }
    }
}
#include <iostream>
#include <getopt.h>
#include <string>

#include "options.hpp"
#include "vevid.cuh"
#include "detect_net.hpp"

using namespace std; 
using namespace cv; 

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "   -p [ARGS...]  : Specify input parameters (separate ARGS with commas)" << endl; 
    cout << "       ARGS: <PARAM=val>, where val is a float" << endl;
    cout << "           PARAM:" << endl;  
    cout << "               S:  Phase strength" << endl; 
    cout << "               T:  Variance of the spectral phase function" << endl; 
    cout << "               b:  Regularization term" << endl; 
    cout << "               G:  Phase activation gain" << endl; 
    cout << endl; 
    cout << "   -r [ARGS...]   : Specify resolution" << endl; 
    cout << "       ARGS: <W>,<H>" << endl; 
    cout << "           W (int):    Image width" << endl; 
    cout << "           H (int):    Image height" << endl; 
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
    while ((c = getopt(argc, argv, ":p:r:i:v:w:ldth")) != -1) {
        specified = true;
        switch (c) {
            case 'p':
                flags->p_value = optarg;
                break;
            case 'r': 
                flags->r_value = optarg; 
                break; 
            case 'i':
                flags->i_value = optarg;
                break;
            case 'v':
                flags->v_value = optarg;
                break;
            case 'w':
                flags->w_value = optarg;
                break;
            case 'l':
                flags->l_flag = true;
                break;
            case 'd':
                flags->d_flag = true;
                break;
            case 't':
                flags->t_flag = true;
                break;
            case 'h':
                flags->h_flag = true;
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

    if (flags->h_flag == true) {
        print_usage(argv[0]);
        exit(0);
    }

    if (flags->t_flag == true) {
        cout << "Timing information will be displayed" << endl;
    }

    if (flags->i_value != nullptr && flags->v_value != nullptr) {
        cout << "both -i and -v flags specified" << endl;
        print_usage(argv[0]);
        exit(0);
    }
    
    if (flags->r_value != nullptr) {
        string input(flags->r_value); 
        istringstream iss(input); 
        string token; 

        int int_count = 0; 

        while (getline(iss, token, ',')) {
            if (int_count < 2) {
                int int_value; 
                try {
                    int_value = stoi(token); 
                }
                catch (const invalid_argument& e) {
                    cout << "Invalid integer value in custom resolution parameters: " << token << endl;
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
            else {
                cout << "Too many custom parameters" << endl;
                print_usage(argv[0]);
                exit(0);
            }
        }
        if (int_count != 2) {
            cout << "Too few custom parameters" << endl;
            print_usage(argv[0]);
            exit(0);
        }

        cout << "Custom resolution parameters specified, using:" << endl;
        cout << "   width = " << params->width << endl;
        cout << "   height = " << params->height << endl;
    }

    if (flags->p_value == nullptr) {
        cout << "Custom parameters not specified, using default values:" << endl;
        cout << "   S = " << params->S << endl;
        cout << "   T = " << params->T << endl;
        cout << "   b = " << params->b << endl;
        cout << "   G = " << params->G << endl;
    }
    else {
        string input(flags->p_value);
        istringstream iss(input);
        string token;

        int int_count = 0;
        int float_count = 0;

        while(getline(iss, token, ',')) {
            
            if (float_count < 4) {
                float float_value;

                if (token[1] != '=') {
                    cout << "Invalid parameter format" << endl;
                    print_usage(argv[0]);
                    exit(0);
                }

                string float_string = token.substr(2);

                try {
                    float_value = stof(float_string);
                }
                catch (const invalid_argument& e) {
                    cout << "Invalid float value in custom parameters: " << float_string << endl;
                    print_usage(argv[0]);
                    exit(0);
                }
                catch (const out_of_range& e) {
                    cout << "Float value out of range: " << float_string << endl; 
                    print_usage(argv[0]); 
                    exit(0); 
                }
                switch (token[0]) {
                    case 'S':
                        params->S = float_value;
                        break;
                    case 'T':
                        params->T = float_value;
                        break;
                    case 'b':
                        params->b = float_value;
                        break;
                    case 'G':
                        params->G = float_value;
                        break;
                    default: 
                        cout << token[0] << " is not a valid parameter" << endl; 
                        print_usage(argv[0]); 
                        exit(0); 
                }
                float_count++;
            }
            else {
                cout << "Too many custom parameters" << endl;
                print_usage(argv[0]);
                exit(0);
            }
        }

        cout << "Custom parameters specified, using:" << endl;
        cout << "   S = " << params->S << endl;
        cout << "   T = " << params->T << endl;
        cout << "   b = " << params->b << endl;
        cout << "   G = " << params->G << endl;
    }
}
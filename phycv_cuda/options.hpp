#ifndef OPTIONS_H
#define OPTIONS_H

struct Flags {
    char* p_value = nullptr; 
    char* i_value = nullptr; 
    char* v_value = nullptr; 
    char* w_value = nullptr; 
    bool l_flag = false; 
    bool d_flag = false; 
    bool t_flag = false; 
    bool h_flag = false; 
};

struct Params {
    int width = 640; 
    int height = 480; 
    float S = 0.8; // S < 1 (0.5 - 1), smaller S, stronger enhancement, decreasing increases noise
    float T = 0.01; // t should be between 0.01 and 0.001
    float b = 0.2; // b should be between 0.1 and 0.4, increasing suppresses noise
    float G = 1; // (inverse gain), smaller G, stronger enhancement, decreasing increases noise
}; 

void print_usage(const char* program_name);
void process_args(int argc, char* argv[], Flags* flags, Params* params);

#endif // OPTIONS_H
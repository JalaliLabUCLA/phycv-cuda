#ifndef DETECT_NET_HPP
#define DETECT_NET_HPP

#include <detectNet.h>
#include <objectTracker.h>
#include <signal.h>

class DetectNet {
public: 
    DetectNet(uchar3* d_image, int width, int height); 
    ~DetectNet(); 
    void create(); 
    void run(cv::Mat& frame);

    float m_box_left; 
    float m_box_top; 
    float m_box_right; 
    float m_box_bottom; 

private: 
    detectNet* m_net;
    uchar3* d_image; 
    int m_width;
    int m_height;  
}; 

#endif // DETECT_NET_HPP
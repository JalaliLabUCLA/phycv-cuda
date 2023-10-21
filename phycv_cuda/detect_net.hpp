#ifndef DETECT_NET_H
#define DETECT_NET_H

#include <detectNet.h>
#include <objectTracker.h>
#include <signal.h>

class DetectNet {
public: 
    DetectNet(uchar3* d_image, int width, int height); 
    ~DetectNet(); 
    void create(); 
    void run(cv::Mat& frame);

private: 
    detectNet* m_net;
    uchar3* d_image; 
    int m_width;
    int m_height;  
}; 

#endif //  DETECT_NET_H
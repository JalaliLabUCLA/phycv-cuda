#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detect_net.hpp"

using namespace cv; 
using namespace std; 

DetectNet::DetectNet(uchar3* image, int width, int height)
: m_width(width), m_height(height), m_box_left(0), m_box_top(0), m_box_right(0), m_box_bottom(0), d_image(image), m_net(nullptr)
{}

void DetectNet::create()
{
    m_net = detectNet::Create("facedetect", 0.5); 
    cudaMalloc((void**)&d_image, m_width * m_height * sizeof(uchar3)); 
}

void DetectNet::run(Mat& frame)
{
    cvtColor(frame, frame, COLOR_BGR2RGB); 
    cudaDeviceSynchronize(); 
    cudaMemcpy2D(d_image, m_width * sizeof(uchar3), frame.data, frame.step, m_width * sizeof(uchar3), m_height, cudaMemcpyHostToDevice);
    detectNet::Detection* detections = NULL;
    const int numDetections = m_net->Detect(d_image, m_width, m_height, &detections);
    cudaDeviceSynchronize(); 
    cvtColor(frame, frame, COLOR_RGB2BGR);

    if( numDetections > 0 )
    {
        LogVerbose("%i objects detected\n", numDetections);
    
        for( int n=0; n < numDetections; n++ )
        {
            LogVerbose("\ndetected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, m_net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
            LogVerbose("bounding box %i  (%.2f, %.2f)  (%.2f, %.2f)  w=%.2f  h=%.2f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height());

            if (n == 0) {
                m_box_left = detections[n].Left; 
                m_box_top = detections[n].Top; 
                m_box_right = detections[n].Right; 
                m_box_bottom = detections[n].Bottom; 
            }
        
            if( detections[n].TrackID >= 0 ) // is this a tracked object?
                LogVerbose("tracking  ID %i  status=%i  frames=%i  lost=%i\n", detections[n].TrackID, detections[n].TrackStatus, detections[n].TrackFrames, detections[n].TrackLost);
        
            rectangle(frame, Point(detections[n].Left, detections[n].Top), Point(detections[n].Right, detections[n].Bottom), Scalar(0, 255, 0), 2);
        }
    }
}

DetectNet::~DetectNet()
{

}
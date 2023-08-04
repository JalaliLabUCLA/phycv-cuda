#include <thread>

#include "video.hpp"

using namespace std; 
using namespace cv; 

int main() 
{
    /* Initialize constants */
    const int device = 0; 
    const int width = 640; 
    const int height = 480; 

    /* Initialize WebCam and Window */
    WebCam webcam(device, width, height); 
    Window window("Original Video", "Vevid-Enhanced Video"); 

    /* Start the capturing thread */
    thread capture_thread(&WebCam::start_capturing, &webcam); 

    /* Start the display thread */
    thread display_thread(&Window::start_display, &window, ref(webcam), true); 

    /* Wait for the display thread to finish */
    display_thread.join(); 

    /* Stop the capturing thread */
    webcam.stop_capturing(); 
    capture_thread.join(); 

    destroyAllWindows(); 
    return 0; 
}

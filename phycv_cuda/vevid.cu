#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "kernels.cuh"

using namespace cv; 
using namespace std; 

struct Parameters {
    float phase_strength; 
    float warp_strength; 
    float spectral_phase_variance;
    float regularization_term; 
    float phase_activation_gain; 

    Parameters() : phase_strength(1), warp_strength(1), spectral_phase_variance(1), 
        regularization_term(1), phase_activation_gain(1)
    {}
};

int frameCount = 0; 
chrono::high_resolution_clock::time_point startTime, endTime;

// Function to display the achieved framerate
void displayFramerate(Mat& image, double fps) {
    string text = "FPS: " + to_string(fps);
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    int thickness = 2;
    int baseline = 0;

    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    Point textOrg(image.cols - textSize.width - 10, image.rows - 10);

    rectangle(image, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0, 0, 0), FILLED);
    putText(image, text, textOrg, fontFace, fontScale, Scalar(255, 255, 255), thickness);
}


int main(int argc, char** argv) {

    //string GSTREAMER_PIPELINE = "nvarguscamerasrc sensor-mode=0   exposuretimerange='1000000 1000000' wbmode=0 gainrange='1 1' ispdigitalgainrange='1 1' tnr-mode=2 tnr-strength=1.0 ee-mode=2 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)21/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink";

    // Set up frame buffer and VideoCapture object
    Mat image;
    VideoCapture cap(CAP_ANY);
    if (!cap.isOpened()) {
        cout << "No video stream detected." << endl;
        system("pause");
        return 1;
    }

    // Initialize and set Frame Width And Height
    const size_t width = 1920; 
    const size_t height = 1080; 
    const size_t N = width * height; 
    cap.set(CAP_PROP_FRAME_WIDTH, width); 
    cap.set(CAP_PROP_FRAME_HEIGHT, height);
    cout << "width: " << width << endl;
    cout << "height: " << height << endl;

    namedWindow("Raw Feed", WINDOW_NORMAL); 
    namedWindow("VEViD", WINDOW_NORMAL); 

    // Calculate the width for each frame display
    int frameWidth = width; 
    int frameHeight = height; 
    int displayWidth = frameWidth * 2;
    int displayHeight = frameHeight; 

    // Create empty matrix to hold combined frames
    //Mat combinedImage(displayHeight, displayWidth, CV_8UC3, Scalar(0,0,0));

    double averageFramerate = 0.0; 

    // Allocate GPU memory
    cufftComplex* d_vevid_kernel;
    cufftComplex* d_image; 
    uint8_t* d_buffer; 
    float* d_max; 
    float* d_min; 
    cudaMalloc((void**)&d_vevid_kernel, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_image, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&d_buffer, N * sizeof(uint8_t)); 
    cudaMalloc((void**)&d_max, sizeof(float)); 
    cudaMalloc((void**)&d_min, sizeof(float)); 

    // Initialize FFT Plans
    cufftHandle plan; 
    cufftPlan2d(&plan, (int)height, (int)width, CUFFT_C2C); // swapped width and height because cufft requires column-major order

    // Set up input parameters
    Parameters params;
    params.phase_strength = 10;
    params.spectral_phase_variance = 0.1; 
    params.regularization_term = 4; 
    params.phase_activation_gain = 2.2; 

    // Measure Host to Device Data Copy Time 
    cudaEvent_t startHtoDCopy, stopHtoDCopy; 
    cudaEventCreate(&startHtoDCopy);
    cudaEventCreate(&stopHtoDCopy);
    float HtoD_time = 0;

    // Measure FFT Time
    cudaEvent_t startFFT, stopFFT; 
    cudaEventCreate(&startFFT); 
    cudaEventCreate(&stopFFT); 
    float FFT_time = 0; 

    // Measure Vevid Kernel Time
    cudaEvent_t startVevid, stopVevid; 
    cudaEventCreate(&startVevid); 
    cudaEventCreate(&stopVevid); 
    float vevid_time = 0; 

    // Measure Max Reduce Kernel Time
    cudaEvent_t startMax, stopMax; 
    cudaEventCreate(&startMax); 
    cudaEventCreate(&stopMax); 
    float max_time = 0; 

    // Measure FFTShift Kernel Time
    cudaEvent_t startFFTShift, stopFFTShift; 
    cudaEventCreate(&startFFTShift); 
    cudaEventCreate(&stopFFTShift); 
    float FFTShift_time = 0;

    // Measure Hadamard Product Time
    cudaEvent_t startHadamard, stopHadamard; 
    cudaEventCreate(&startHadamard); 
    cudaEventCreate(&stopHadamard); 
    float hadamard_time = 0; 

    // Measure IFFT Time
    cudaEvent_t startIFFT, stopIFFT; 
    cudaEventCreate(&startIFFT); 
    cudaEventCreate(&stopIFFT); 
    float IFFT_time = 0; 

    // Measure Phase Time
    cudaEvent_t startPhase, stopPhase; 
    cudaEventCreate(&startPhase); 
    cudaEventCreate(&stopPhase); 
    float phase_time = 0;

    // Measure Device to Host Data Copy Time
    cudaEvent_t startDtoHCopy, stopDtoHCopy; 
    cudaEventCreate(&startDtoHCopy); 
    cudaEventCreate(&stopDtoHCopy); 
    float DtoH_time = 0; 

    // Measure CPU Operations Time
    float read_time = 0; 
    float BGRtoHSV_time = 0; 
    float merge_time = 0; 
    float HSVtoBGR_time = 0; 

    // Measure Overall Time
    startTime = chrono::high_resolution_clock::now();

    // Set up while loop to read video frames into buffer
    while (true) {

        // Read frame into buffer
        auto startRead = chrono::high_resolution_clock::now(); 
	    cap >> image; 
        auto endRead = chrono::high_resolution_clock::now(); 
        chrono::duration<float> read_frame = chrono::duration_cast<chrono::duration<float>>(endRead - startRead);
        read_time += read_frame.count(); 

	    // Display unaltered video feed (for reference)
        imshow("Raw Feed", image);

        // Convert from BGR to HSV 
        auto startBGRtoHSV = chrono::high_resolution_clock::now(); 
        cvtColor(image, image, COLOR_BGR2HSV); 
        auto endBGRtoHSV = chrono::high_resolution_clock::now(); 
        chrono::duration<float> BGRtoHSV_frame = chrono::duration_cast<chrono::duration<float>>(endBGRtoHSV - startBGRtoHSV);
        BGRtoHSV_time += BGRtoHSV_frame.count(); 

        // Split channels of HSV matrix
        vector<Mat> hsv_channels; 
        split(image, hsv_channels);  

        // Get pointer to V channel of HSV matrix
        uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0);

        // --Start of Algorithm Code--
        // Copy data from host to device
        cudaEventRecord(startHtoDCopy); 
        cudaMemcpy(d_buffer, idata, N * sizeof(uint8_t), cudaMemcpyHostToDevice); 
        cudaEventRecord(stopHtoDCopy); 
        cudaEventSynchronize(stopHtoDCopy); 
        float HtoD_frame = 0; 
        cudaEventElapsedTime(&HtoD_frame, startHtoDCopy, stopHtoDCopy); 
        HtoD_time += HtoD_frame; 


        // Call kernels
        int block_size = 32; 
        int grid_size = ((int)N + block_size - 1) / block_size;

        // Take FFT
        cudaEventRecord(startFFT); 
        populate_real <<<grid_size, block_size >>> (d_image, d_buffer, N); 
        add <<<grid_size, block_size>>> (d_image, params.regularization_term, N);     
        cufftExecC2C(plan, d_image, d_image, CUFFT_FORWARD); 
        cudaEventRecord(stopFFT); 
        cudaEventSynchronize(stopFFT); 
        float FFT_frame = 0; 
        cudaEventElapsedTime(&FFT_frame, startFFT, stopFFT); 
        FFT_time += FFT_frame; 

        // Compute Kernel
        cudaEventRecord(startVevid);
        vevid_kernel <<<grid_size, block_size>>> (d_vevid_kernel, params.phase_strength, params.spectral_phase_variance, width, height);
        cudaEventRecord(stopVevid);
        cudaEventSynchronize(stopVevid); 
        float vevid_frame = 0; 
        cudaEventElapsedTime(&vevid_frame, startVevid, stopVevid); 
        vevid_time += vevid_frame; 
        
        cudaEventRecord(startMax);
        max_reduce <<<64, block_size, block_size * sizeof(float)>>> (d_vevid_kernel, d_max, N);
        cudaEventRecord(stopMax); 
        cudaEventSynchronize(stopMax); 
        float max_frame = 0; 
        cudaEventElapsedTime(&max_frame, startMax, stopMax); 
        max_time += max_frame; 

        cudaEventRecord(startFFTShift); 
        float max_val; 
        cudaMemcpy(&max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost); 
	    scale <<<grid_size, block_size>>> (d_vevid_kernel, (1.0f / max_val), N); 
        fftshift <<<grid_size, block_size>>> (d_vevid_kernel, width, height); 
        cudaEventRecord(stopFFTShift); 
        cudaEventSynchronize(stopFFTShift); 
        float FFTShift_frame = 0; 
        cudaEventElapsedTime(&FFTShift_frame, startFFTShift, stopFFTShift); 
        FFTShift_time += FFTShift_frame; 

        // Multiply image with vevid kernel in frequency domain
        cudaEventRecord(startHadamard); 
        hadamard <<<grid_size, block_size>>> (d_vevid_kernel, d_image, N); 
        cudaEventRecord(stopHadamard); 
        cudaEventSynchronize(stopHadamard); 
        float hadamard_frame = 0; 
        cudaEventElapsedTime(&hadamard_frame, startHadamard, stopHadamard); 
        hadamard_time += hadamard_frame; 

        // Take IFFT
        cudaEventRecord(startIFFT); 
        cufftExecC2C(plan, d_image, d_image, CUFFT_INVERSE); 
        scale <<<grid_size, block_size>>> (d_image, (1.0f / (float)N), N); 
        cudaEventRecord(stopIFFT); 
        cudaEventSynchronize(stopIFFT); 
        float IFFT_frame = 0; 
        cudaEventElapsedTime(&IFFT_frame, startIFFT, stopIFFT); 
        IFFT_time += IFFT_frame; 

        // Get vevid phase
        cudaEventRecord(startPhase); 
        vevid_phase <<<grid_size, block_size>>> (d_image, d_buffer, params.phase_activation_gain, N); 
        cudaEventRecord(stopPhase); 
        cudaEventSynchronize(stopPhase); 
        float phase_frame = 0; 
        cudaEventElapsedTime(&phase_frame, startPhase, stopPhase); 
        phase_time += phase_frame; 
	    // --End of Algorithm Code

        // Copy data from device to host
        cudaEventRecord(startDtoHCopy); 
        cudaMemcpy(idata, d_buffer, N * sizeof(uint8_t), cudaMemcpyDeviceToHost); 
        cudaEventRecord(stopDtoHCopy); 
        cudaEventSynchronize(stopDtoHCopy); 
        float DtoH_frame = 0; 
        cudaEventElapsedTime(&DtoH_frame, startDtoHCopy, stopDtoHCopy); 
        DtoH_time += DtoH_frame; 
       	
        auto startMerge = chrono::high_resolution_clock::now(); 
        merge(hsv_channels, image); 
        auto endMerge = chrono::high_resolution_clock::now(); 
        chrono::duration<float> merge_frame = chrono::duration_cast<chrono::duration<double>>(endMerge - startMerge); 
        merge_time += merge_frame.count(); 

        // Convert from HSV to BGR
        auto startHSVtoBGR = chrono::high_resolution_clock::now(); 
        cvtColor(image, image, COLOR_HSV2BGR); 
        auto endHSVtoBGR = chrono::high_resolution_clock::now(); 
        chrono::duration<float> HSVtoBGR_frame = chrono::duration_cast<chrono::duration<double>>(endHSVtoBGR - startHSVtoBGR); 
        HSVtoBGR_time += HSVtoBGR_frame.count();

        // Calculate and display the achieved framerate
        frameCount++; 
        endTime = chrono::high_resolution_clock::now(); 
        chrono::duration<double> duration = chrono::duration_cast<chrono::duration<double>>(endTime - startTime);
        double currentFramerate = frameCount / duration.count();
        averageFramerate = (averageFramerate * (frameCount - 1) + currentFramerate) / frameCount;
        displayFramerate(image, averageFramerate);

        // Display Processed video feed
        imshow("VEViD", image); 

        // Exit on escape key
        char c = (char)waitKey(1);
        if (c == 27) {
            float averageHtoDCopyTimePerFrame = HtoD_time / frameCount; 
            float averageFFTTimePerFrame = FFT_time / frameCount; 
            float averageVevidTimePerFrame = vevid_time / frameCount; 
            float averageMaxTimePerFrame = max_time / frameCount; 
            float averageFFTShiftTimePerFrame = FFTShift_time / frameCount; 
            float averageHadamardTimePerFrame = hadamard_time / frameCount; 
            float averageIFFTTimePerFrame = IFFT_time / frameCount; 
            float averagePhaseTimePerFrame = phase_time / frameCount; 
            float averageDtoHCopyTimePerFrame = DtoH_time / frameCount; 

            float averageReadTimePerFrame = read_time /frameCount; 
            float averageBGRtoHSVTimePerFrame = BGRtoHSV_time / frameCount; 
            float averageMergeTimePerFrame = merge_time / frameCount; 
            float averageHSVtoBGRTimePerFrame = HSVtoBGR_time / frameCount; 

            cout << "Frames Captured: " << frameCount << endl; 
            cout << "--Gpu Operations--" << endl; 
            cout << "Average Host to Device Copy Time Per Frame: " << averageHtoDCopyTimePerFrame << " ms" << endl; 
            cout << "Average FFT Time Per Frame: " << averageFFTTimePerFrame << " ms" << endl; 
            cout << "Average Vevid Kernel Time Per Frame: " << averageVevidTimePerFrame << " ms" << endl; 
            cout << "Average Max Reduce Time Per Frame: " << averageMaxTimePerFrame << " ms" << endl; 
            cout << "Average FFTShift Time Per Frame: " << averageFFTShiftTimePerFrame << " ms" << endl; 
            cout << "Average Hadamard Product Time Per Frame: " << averageHadamardTimePerFrame << " ms" << endl; 
            cout << "Average IFFT Time Per Frame: " << averageIFFTTimePerFrame << " ms" << endl; 
            cout << "Average Phase Time Per Frame: " << averagePhaseTimePerFrame << " ms" << endl; 
            cout << "Average Device to Host Copy Time Per Frame: " << averageDtoHCopyTimePerFrame << " ms" << endl; 
            cout << "--CPU Operations--" << endl; 
            cout << "Average Read Time Per Frame: " << averageReadTimePerFrame * 1000 << " ms" << endl; 
            cout << "Average BGR to HSV Conversion Time Per Frame: " << averageBGRtoHSVTimePerFrame * 1000 << " ms" << endl; 
            cout << "Average Merge Time Per Frame: " << averageMergeTimePerFrame * 1000 << " ms" << endl; 
            cout << "Average HSV to BGR Conversion Time Per Frame: " << averageHSVtoBGRTimePerFrame * 1000 << " ms" << endl; 

            break;
        }
    }
    // Free frame buffer
    cap.release(); 

    // Free GPU memory
    cudaFree(d_vevid_kernel); 
    cudaFree(d_image); 
    cudaFree(d_buffer);
    cudaFree(d_max); 
    cudaFree(d_min); 

    return 0; 
}



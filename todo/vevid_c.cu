struct Parameters { // TODO: change to b,G,S,T, explain meaning once
    float phase_strength; 
    float warp_strength; 
    float spectral_phase_variance;
    float regularization_term; 
    float phase_activation_gain; 

    Parameters() : phase_strength(1), warp_strength(1), spectral_phase_variance(1), 
        regularization_term(1), phase_activation_gain(1)
    {}
};

struct Times {
    float BGR_to_HSV_time; 
    float data_in_time; 
    float fft_time; 
    float kernel_multiplication_time; 
    float ifft_time; 
    float phase_time;
    float data_out_time; 
    float HSV_to_BGR_time;  
    float vevid_time; 
}; 

struct VevidContext
{
    /* GPU resources */
    cufftComplex *d_vevid_kernel; 
    cufftComplex *d_image;
    uint8_t *d_buffer; 
    float *d_max; 
    float *d_min;
    float *d_phase_min; 
    float *d_phase_max; 

    uchar3* d_rgb; 
    float3* d_hsv; 

    /* FFT plan */ 
    cufftHandle plan; 

    /* Input parameters */
    Parameters params; 

    /* Frame width and height */
    size_t width; 
    size_t height; 
}; 

VevidContext context; 
Times times; 

void vevid_init(int width, int height,
    float phase_strength, 
    float spectral_phase_variance, 
    float regularization_term, 
    float phase_activation_gain) 
{
    context.width = width; 
    context.height = height; 
    const size_t N = width * height; 

    /* Allocate GPU memory */
    cudaMalloc((void**)&context.d_vevid_kernel, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&context.d_image, N * sizeof(cufftComplex)); 
    cudaMalloc((void**)&context.d_buffer, N * sizeof(uint8_t)); 
    cudaMalloc((void**)&context.d_max, sizeof(float)); 
    cudaMalloc((void**)&context.d_min, sizeof(float)); 
    cudaMalloc((void**)&context.d_phase_max, sizeof(float)); 
    cudaMalloc((void**)&context.d_phase_min, sizeof(float)); 

    cudaMalloc((void**)&context.d_rgb, N * sizeof(uchar3)); 
    cudaMalloc((void**)&context.d_hsv, N * sizeof(float3)); 

    /* Initialize FFT plan */
    cufftPlan2d(&context.plan, (int)height, (int)width, CUFFT_C2C); 

    /* Set up input parameters */
    context.params.phase_strength = phase_strength; 
    context.params.spectral_phase_variance = spectral_phase_variance; 
    context.params.regularization_term = regularization_term; 
    context.params.phase_activation_gain = phase_activation_gain; 

    int block_size = 32; 
    int grid_size = ((int)N + block_size - 1) / block_size; 

    /* Compute VEViD kernel */
    vevid_kernel<<<grid_size, block_size>>>(context.d_vevid_kernel, context.params.phase_strength, context.params.spectral_phase_variance, width, height);
    max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_vevid_kernel, context.d_max, N);
    float max_val; 
    cudaMemcpy(&max_val, context.d_max, sizeof(float), cudaMemcpyDeviceToHost);  
    scale<<<grid_size, block_size>>>(context.d_vevid_kernel, (1.0f / max_val), N);
    fftshift<<<grid_size, block_size>>>(context.d_vevid_kernel, width, height);
}

void vevid(cv::Mat& image, bool show_timing, bool lite) {
    auto vevid_start = chrono::high_resolution_clock::now(); 

    size_t width = context.width; 
    size_t height = context.height; 
    size_t N = width * height; 

    /* Convert from BGR to HSV */
    float to_HSV_time; 
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_BGR2HSV), to_HSV_time);
    
    /*
    cvtColor(image, image, COLOR_BGR2RGB); 
    uchar3* rgb = image.ptr<uchar3>(0); 
    uchar3* out; 
    cudaMemcpy(context.d_rgb, rgb, N * sizeof(uchar3), cudaMemcpyHostToDevice); 
    convert_to_hsv_wrapper(context.d_rgb, context.d_hsv, width, height); 
    */

    /* Get pointer to V channel of HSV matrix */
    auto start = chrono::high_resolution_clock::now(); 
    vector<Mat> hsv_channels; 
    split(image, hsv_channels); 
    uint8_t* idata = hsv_channels[2].ptr<uint8_t>(0); 
    auto stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, std::milli> elapsed = stop - start; 
    times.BGR_to_HSV_time = to_HSV_time + elapsed.count(); 

    /* -- Start of Algorithm Code -- */
    /* Copy data from host to device */
    MEASURE_GPU_TIME(cudaMemcpy(context.d_buffer, idata, N * sizeof(uint8_t), cudaMemcpyHostToDevice), times.data_in_time); 

    /* Call CUDA kernels */
    int block_size = 32; 
    int grid_size = ((int)N + block_size - 1) / block_size; 

    if (lite) {
        vevid_phase_lite<<<grid_size, block_size>>>(context.d_buffer, context.params.phase_activation_gain, context.params.regularization_term, N); 
    }
    else {
        /* Take FFT */
        float populate_real_time; 
        float add_time; 
        float forward_fft_time; 
        MEASURE_GPU_TIME((populate_real<<<grid_size, block_size>>>(context.d_image, context.d_buffer, N)), populate_real_time);
        MEASURE_GPU_TIME((add<<<grid_size, block_size>>>(context.d_image, context.params.regularization_term, N)), add_time);
        MEASURE_GPU_TIME(cufftExecC2C(context.plan, context.d_image, context.d_image, CUFFT_FORWARD), forward_fft_time);
        times.fft_time = populate_real_time + add_time + forward_fft_time;

        /* Multiply kernel with image in frequency domain */
        float hadamard_time; 
        MEASURE_GPU_TIME((hadamard<<<grid_size, block_size>>>(context.d_vevid_kernel, context.d_image, N)), hadamard_time);
        times.kernel_multiplication_time = hadamard_time; 

        /* Take IFFT */
        float backward_fft_time; 
        float scale_time; 
        MEASURE_GPU_TIME(cufftExecC2C(context.plan, context.d_image, context.d_image, CUFFT_INVERSE), backward_fft_time);
        MEASURE_GPU_TIME((scale<<<grid_size, block_size>>>(context.d_image, (1.0f / (float)N), N)), scale_time);
        times.ifft_time = backward_fft_time + scale_time; 

        /* Get phase */
        float phase_time; 
        MEASURE_GPU_TIME((vevid_phase<<<grid_size, block_size>>>(context.d_image, context.d_buffer, context.params.phase_activation_gain, N)), phase_time);
        //max_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_image, context.d_phase_max, N);
        //min_reduce<<<64, block_size, block_size * sizeof(float)>>>(context.d_image, context.d_phase_min, N);
        //float max_phase;
        //float min_phase;  
        //cudaMemcpy(&max_phase, context.d_phase_max, sizeof(float), cudaMemcpyDeviceToHost); 
        //cudaMemcpy(&min_phase, context.d_phase_min, sizeof(float), cudaMemcpyDeviceToHost); 
        //vevid_normalize<<<grid_size, block_size>>>(context.d_image, context.d_buffer, 0, -M_PI / 2, N);
        //cout << max_phase << ", " << min_phase << endl; 
        times.phase_time = phase_time; 
    }

    /* Copy data from device to host */
    MEASURE_GPU_TIME(cudaMemcpy(idata, context.d_buffer, N * sizeof(uint8_t), cudaMemcpyDeviceToHost), times.data_out_time); 
    /* -- End of Algorithm Code -- */

    /* Merge channels */
    float merge_time; 
    float to_BGR_time; 
    MEASURE_CPU_TIME(merge(hsv_channels, image), merge_time); 

    /* Normalize image */
    //cv::normalize(image, image, 0, 255, NORM_MINMAX); 

    /* Convert from HSV to BGR */
    MEASURE_CPU_TIME(cvtColor(image, image, COLOR_HSV2BGR), to_BGR_time);
    times.HSV_to_BGR_time = merge_time + to_BGR_time; 

    auto vevid_stop = chrono::high_resolution_clock::now(); 
    chrono::duration<float, milli> total = vevid_stop - vevid_start; 
    times.vevid_time = total.count(); 

    if (show_timing) {
        cout << "Timing Results: " << endl; 
        cout << "BGR to HSV time per frame: " << times.BGR_to_HSV_time << " ms" << endl; 
        cout << "Forward data copy time: " << times.data_in_time << " ms" << endl; 
        cout << "FFT time per frame: " << times.fft_time << " ms" << endl; 
        cout << "Kernel multiplication time per frame: " << times.kernel_multiplication_time << " ms" << endl; 
        cout << "Inverse FFT time per frame: " << times.ifft_time << " ms" << endl; 
        cout << "Phase time per frame: " << times.phase_time << " ms" << endl; 
        cout << "Backward data copy time per frame: " << times.data_out_time << " ms" << endl; 
        cout << "HSV to BGR time per frame: " << times.HSV_to_BGR_time << " ms" << endl; 
        cout << "Total time per frame: " << times.vevid_time << " ms" << endl; 
        cout << endl; 
    }
}

void vevid_fini() {
    /* Free GPU memory */
    cudaFree(context.d_vevid_kernel); 
    cudaFree(context.d_image); 
    cudaFree(context.d_buffer); 
    cudaFree(context.d_max); 
    cudaFree(context.d_min); 

    /* Destroy FFT plan */
    cufftDestroy(context.plan); 
}
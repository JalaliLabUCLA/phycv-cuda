#ifndef VEVID_CUH
#define VEVID_CUH

#include <opencv2/core.hpp>

void vevid_init(int width, int height, float phase_strength, float spectral_phase_variance, float regularization_term, float phase_activation_gain);
void vevid(cv::Mat& image, bool show_timing, bool lite);
void vevid_fini();

#endif // VEVID_H

#pragma once

#include "MLFeaturesManager.h"

#if ENABLE_IMAGE_FEATURES_EXTRACTION

#include <opencv2/opencv.hpp>
#include <array>
#include <tuple>

struct HadamardFeatures {
    double dc, energy_total, energy_ac, max_coef, min_coef, top_left, top_right, bottom_left, bottom_right;
};

void fwht_1d(cv::Mat& vec);
cv::Mat fwht_2d(const cv::Mat& blk);
std::tuple<double,double,double,double> calculate_basic_features_cv(const cv::Mat& blk);
std::array<double,4> calculate_stats_cv(const cv::Mat& blk);
std::array<double,5> calculate_gradients_sobel_cv(const cv::Mat& blk);
std::array<double,5> calculate_gradients_prewitt_cv(const cv::Mat& blk);
std::array<double,3> calculate_contrast_features_cv(const cv::Mat& blk);
double calculate_laplacian_var_cv(const cv::Mat& blk);
double calculate_entropy_cv(const cv::Mat& blk);
HadamardFeatures calculate_hadamard_features(const cv::Mat& blk);

bool computeImageFeatures(const vvenc::Pel* buf, int stride, int width, int height, MLFeatureData& data);

#endif
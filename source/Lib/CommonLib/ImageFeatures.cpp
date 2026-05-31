#include "ImageFeatures.h"

#if ENABLE_IMAGE_FEATURES_EXTRACTION

// 1D Fast Walsh-Hadamard Transform (in-place) - REMOVED
/*
void fwht_1d(cv::Mat& vec) {
    int n = vec.cols > 1 ? vec.cols : vec.rows;
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = vec.at<float>(i + j);
                float v = vec.at<float>(i + j + len);
                
                vec.at<float>(i + j) = u + v;
                vec.at<float>(i + j + len) = u - v;
            }
        }
    }
}
*/

// 2D Hadamard Transform - REMOVED
/*
cv::Mat fwht_2d(const cv::Mat& blk) {
    cv::Mat H;
    blk.convertTo(H, CV_32F);

    for (int r = 0; r < H.rows; r++) {
        cv::Mat row = H.row(r);
        fwht_1d(row);
    }

    for (int c = 0; c < H.cols; c++) {
        cv::Mat col = H.col(c);
        fwht_1d(col);
    }

    return H;
}
*/

// Mean, Variance, StdDev and Sum
std::tuple<double,double,double,double> calculate_basic_features_cv(const cv::Mat& blk) {
    cv::Mat blk_f;
    blk.convertTo(blk_f, CV_32F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(blk_f, mean, stddev);

    double var = stddev[0] * stddev[0];
    double sum = cv::sum(blk_f)[0];

    return std::make_tuple(mean[0], var, stddev[0], sum);
}

// vH, vV, dH, dV
std::array<double,4> calculate_stats_cv(const cv::Mat& blk) {
    cv::Mat blk_f;
    blk.convertTo(blk_f, CV_32F);
    
    cv::Mat row_means; 
    cv::reduce(blk_f, row_means, 1, cv::REDUCE_AVG);
    
    cv::Mat row_means_exp; 
    cv::repeat(row_means, 1, blk_f.cols, row_means_exp);
    
    cv::Mat diff_row = blk_f - row_means_exp;
    
    cv::Mat row_vars; 
    cv::reduce(diff_row.mul(diff_row), row_vars, 1, cv::REDUCE_AVG);
    
    cv::Mat row_stds; 
    cv::sqrt(row_vars, row_stds);
    
    double vH = cv::mean(row_vars)[0];
    double dH = cv::mean(row_stds)[0];

    cv::Mat col_means; 
    cv::reduce(blk_f, col_means, 0, cv::REDUCE_AVG);
    
    cv::Mat col_means_exp; 
    cv::repeat(col_means, blk_f.rows, 1, col_means_exp);
    
    cv::Mat diff_col = blk_f - col_means_exp;
    
    cv::Mat col_vars; 
    cv::reduce(diff_col.mul(diff_col), col_vars, 0, cv::REDUCE_AVG);
    
    cv::Mat col_stds; 
    cv::sqrt(col_vars, col_stds);
    
    double vV = cv::mean(col_vars)[0];
    double dV = cv::mean(col_stds)[0];
    
    return {vH, vV, dV, dH};
}

// Sobel Gradients - REMOVED
/*
std::array<double,5> calculate_gradients_sobel_cv(const cv::Mat& blk) {
    cv::Mat blk_f; 
    blk.convertTo(blk_f, CV_32F); 
    
    cv::Mat Gh, Gv;
    cv::Sobel(blk_f, Gh, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(blk_f, Gv, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
    
    double mGv = cv::mean(cv::abs(Gv))[0]; 
    double mGh = cv::mean(cv::abs(Gh))[0];
    
    cv::Mat Mag, Dir; 
    cv::magnitude(Gv, Gh, Mag); 
    cv::phase(Gh, Gv, Dir, true);
    
    double meanMag = cv::mean(Mag)[0]; 
    double meanDir = cv::mean(Dir)[0];
    double razao_grad = mGh / (mGv + 1e-6);
    
    return {mGv, mGh, meanMag, meanDir, razao_grad};
}
*/

// Prewitt Gradients - REMOVED
/*
std::array<double,5> calculate_gradients_prewitt_cv(const cv::Mat& blk) {
    cv::Mat blk_f; 
    blk.convertTo(blk_f, CV_32F);
    
    cv::Mat kernel_gx = (cv::Mat_<float>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat kernel_gy = (cv::Mat_<float>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    
    cv::Mat Gh, Gv;
    cv::filter2D(blk_f, Gh, CV_32F, kernel_gx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(blk_f, Gv, CV_32F, kernel_gy, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    
    double mGv = cv::mean(cv::abs(Gv))[0]; 
    double mGh = cv::mean(cv::abs(Gh))[0];
    
    cv::Mat Mag, Dir; 
    cv::magnitude(Gv, Gh, Mag); 
    cv::phase(Gh, Gv, Dir, true);
    
    double meanMag = cv::mean(Mag)[0]; 
    double meanDir = cv::mean(Dir)[0];
    double razao_grad = mGh / (mGv + 1e-6);
    
    return {mGv, mGh, meanMag, meanDir, razao_grad};
}
*/

// Contrast
std::array<double,3> calculate_contrast_features_cv(const cv::Mat& blk) {
    double minVal, maxVal; 
    cv::minMaxLoc(blk, &minVal, &maxVal);
    
    return {minVal, maxVal, maxVal - minVal};
}

// Sharpness (Laplacian variance) - REMOVED
/*
double calculate_laplacian_var_cv(const cv::Mat& blk) {
    cv::Mat blk_f; 
    blk.convertTo(blk_f, CV_32F); 
    
    cv::Mat lap;
    cv::Laplacian(blk_f, lap, CV_32F, 1, 1, 0);
    
    cv::Scalar mean, stddev; 
    cv::meanStdDev(lap, mean, stddev);     
    
    return stddev[0] * stddev[0];
}
*/

// Shannon Entropy - REMOVED
/*
double calculate_entropy_cv(const cv::Mat& blk) {
    int histSize = 256; 
    float range[] = {0, 256}; 
    const float* histRange = {range};
    
    cv::Mat hist; 
    cv::calcHist(&blk, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    hist /= cv::sum(hist)[0];
    double entropy = 0.0;
    
    for(int i = 0; i < histSize; i++) {
        float p = hist.at<float>(i);
        if(p > 0) {
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}
*/

// Hadamard - REMOVED
/*
HadamardFeatures calculate_hadamard_features(const cv::Mat& blk) {
    cv::Mat H = fwht_2d(blk); 
    HadamardFeatures f{};
    
    f.dc = H.at<float>(0,0); 
    f.energy_total = cv::sum(H.mul(H))[0]; 
    f.energy_ac = f.energy_total - f.dc * f.dc;
    
    double minVal, maxVal; 
    cv::minMaxLoc(H, &minVal, &maxVal);
    
    f.min_coef = minVal; 
    f.max_coef = maxVal;
    
    f.top_left = H.at<float>(0,0); 
    f.top_right = H.at<float>(0, H.cols - 1);
    f.bottom_left = H.at<float>(H.rows - 1, 0); 
    f.bottom_right = H.at<float>(H.rows - 1, H.cols - 1);
    
    return f;
}
*/

bool computeImageFeatures(const vvenc::Pel* buf, int stride, int width, int height, MLFeatureData& data) {
    if (buf == nullptr || width <= 0 || height <= 0) {
        return false; 
    }

    cv::Mat blk(height, width, CV_32F);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            blk.at<float>(y, x) = (float)buf[y * stride + x];
        }
    }
    
    double mean, var, std_dev, sum_val;
    std::tie(mean, var, std_dev, sum_val) = calculate_basic_features_cv(blk);
    data.blkPixelMean = mean; 
    data.blkPixelVariance = var; 
    data.blkPixelStdDev = std_dev; 
    data.blkPixelSum = sum_val;

    auto stats = calculate_stats_cv(blk);
    data.blkVarH = stats[0]; 
    data.blkVarV = stats[1]; 
    data.blkStdV = stats[2]; 
    data.blkStdH = stats[3]; 

    // Removed complex features
    // auto sob = calculate_gradients_sobel_cv(blk);
    // data.blkSobelGv = sob[0]; 
    // data.blkSobelGh = sob[1]; 
    // data.blkSobelMag = sob[2]; 
    // data.blkSobelDir = sob[3]; 
    // data.blkSobelRazaoGrad = sob[4];
    // auto pre = calculate_gradients_prewitt_cv(blk);
    // data.blkPrewittGv = pre[0]; 
    // data.blkPrewittGh = pre[1]; 
    // data.blkPrewittMag = pre[2]; 
    // data.blkPrewittDir = pre[3]; 
    // data.blkPrewittRazaoGrad = pre[4];

    auto contrast = calculate_contrast_features_cv(blk);
    data.blkMin = contrast[0]; 
    data.blkMax = contrast[1]; 
    data.blkRange = contrast[2];

    // Removed complex image features
    // data.blkLaplacianVar = calculate_laplacian_var_cv(blk);
    // data.blkEntropy = calculate_entropy_cv(blk);
    // HadamardFeatures had = calculate_hadamard_features(blk);
    // data.blkHadDc = had.dc; 
    // data.blkHadEnergyTotal = had.energy_total; 
    // data.blkHadEnergyAc = had.energy_ac; 
    // data.blkHadMax = had.max_coef; 
    // data.blkHadMin = had.min_coef;
    // data.blkHadTopLeft = had.top_left; 
    // data.blkHadTopRight = had.top_right; 
    // data.blkHadBottomLeft = had.bottom_left; 
    // data.blkHadBottomRight = had.bottom_right;

    // --- Geometry feature calculation ---
    if (width == height) {
        data.orientationGroup = 0;  // Square
    } else if (width > height) {
        data.orientationGroup = 1;  // Horizontal
    } else {
        data.orientationGroup = 2;  // Vertical
    }
    
    int minDim = std::min(width, height);
    int maxDim = std::max(width, height);
    int ratio = maxDim / std::max(1, minDim);
    data.aspectRatioGroup = (ratio > 0) ? static_cast<int>(std::log2(ratio) + 0.5) : 0;

    // --- New derived feature calculations ---
    
    int fw = MLFeaturesManager::getFrameWidth();
    int fh = MLFeaturesManager::getFrameHeight();
    
    // Relative block area
    if (fw > 0 && fh > 0) {
        data.relativeBlockArea = (double)data.blockArea / (double)(fw * fh);
        
        // Normalized center coordinates
        double centerX = fw / 2.0;
        double centerY = fh / 2.0;
        data.distCenterX = std::abs(data.xPos + (width / 2.0) - centerX) / centerX;
        data.distCenterY = std::abs(data.yPos + (height / 2.0) - centerY) / centerY;
        data.centerFocusWeight = 1.0 / (std::sqrt(std::pow(data.distCenterX, 2) + std::pow(data.distCenterY, 2)) + 1.0);
    } else {
        data.relativeBlockArea = 0.0;
        data.distCenterX = 0.0;
        data.distCenterY = 0.0;
        data.centerFocusWeight = 0.0;
    }

    // Delta QP
    data.deltaQP = data.cuQp - MLFeaturesManager::getTargetQP();

    // Contrast ratio
    data.contrastRatio = data.blkRange / (data.blkPixelMean + 1.0);

    // Directional dominance
    data.directionalDominance = std::abs(data.blkVarH - data.blkVarV) / (data.blkVarH + data.blkVarV + 1.0);

    // Variance per area
    if (data.blockArea > 0) {
        data.variancePerArea = data.blkPixelVariance / (double)data.blockArea;
    } else {
        data.variancePerArea = 0.0;
    }

    // Mean and variance mismatch
    data.meanMismatch = std::abs(data.blkPixelMean - data.refLineMean);
    data.varMismatch = std::abs(data.blkPixelVariance - data.refLineVariance);

    // Coefficient of variation
    data.coefVariation = data.blkPixelStdDev / (data.blkPixelMean + 1.0);

    // Reference dominance (above vs left)
    data.refDominance = data.refLineVariance / (data.refColVariance + 1.0);

    // MPM delta (if both neighbors are intra, otherwise -1)
    if (data.leftIsIntra && data.aboveIsIntra) {
        data.mpmDelta = std::abs(data.leftIntraDir - data.aboveIntraDir);
    } else {
        data.mpmDelta = -1;
    }

    // Boundary complexity ratio
    data.boundaryComplexityRatio = (data.refLineVariance + data.refColVariance) / (data.blkPixelVariance + 1.0);

    // Splitting density (generic VVC max depth is about 6)
    data.splittingDensity = (data.cuMtDepth + data.cuBtDepth) / 6.0;

    return true;
}

#endif

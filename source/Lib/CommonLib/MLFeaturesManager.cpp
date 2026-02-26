#include "MLFeaturesManager.h"
#include "ImageFeatures.h"

std::ofstream MLFeaturesManager::featFp;
std::mutex MLFeaturesManager::writeMutex;

std::string MLFeaturesManager::videoName;
std::string MLFeaturesManager::encoderPreset;
int MLFeaturesManager::targetQP;
int MLFeaturesManager::bitDepth;
int MLFeaturesManager::frameWidth;
int MLFeaturesManager::frameHeight;

thread_local std::vector<MLFeatureData> localBuffer;
const size_t MAX_BUFFER_SIZE = 5000; 

void MLFeaturesManager::init(const std::string& fileName, const std::string& vName, 
                             const std::string& preset, int tQp, int bDepth, int fWidth, int fHeight) {
    videoName = vName;
    encoderPreset = preset;
    targetQP = tQp;
    bitDepth = bDepth;
    frameWidth = fWidth;
    frameHeight = fHeight;

    featFp.open(fileName);
    if (featFp.is_open()) {
        featFp << "VideoName;"
               << "EncoderPreset;"
               << "TargetQP;"
               << "BitDepth;"
               << "FrameWidth;"
               << "FrameHeight;"
               << "Frame;"
               << "X_Pos;"
               << "Y_Pos;"
               << "BlockWidth;"
               << "BlockHeight;"
               << "BlockArea;"
               << "BlockAreaGroup;"
               << "FrameLevel;"
               << "BorderContactMask;"
               << "IsIntra;"
               << "CU_QP;"
               << "Depth;"
               << "QTDepth;"
               << "BTDepth;"
               << "SplitSeries";
        
#if ENABLE_IMAGE_FEATURES_EXTRACTION == 1
        featFp << ";"
               << "blk_pixel_mean;"
               << "blk_pixel_variance;"
               << "blk_pixel_std_dev;"
               << "blk_pixel_sum;"
               << "blk_var_h;"
               << "blk_var_v;"
               << "blk_std_h;"
               << "blk_std_v;"
               << "blk_min;"
               << "blk_max;"
               << "blk_range;"
               << "blk_laplacian_var;"
               << "blk_entropy;"
               << "blk_sobel_gv;"
               << "blk_sobel_gh;"
               << "blk_sobel_mag;"
               << "blk_sobel_dir;"
               << "blk_sobel_razao_grad;"
               << "blk_prewitt_gv;"
               << "blk_prewitt_gh;"
               << "blk_prewitt_mag;"
               << "blk_prewitt_dir;"
               << "blk_prewitt_razao_grad;"
               << "blk_had_dc;"
               << "blk_had_energy_total;"
               << "blk_had_energy_ac;"
               << "blk_had_max;"
               << "blk_had_min;"
               << "blk_had_topleft;"
               << "blk_had_topright;"
               << "blk_had_bottomleft;"
               << "blk_had_bottomright";
#endif
        featFp << "\n";
    } else {
        std::cerr << "[Error] MLFeaturesManager: Could not open file " << fileName << std::endl;
    }
}

void MLFeaturesManager::flushBuffer() {
    if (localBuffer.empty()) return;

    std::lock_guard<std::mutex> lock(writeMutex);
    
    if (featFp.is_open()) {
        for (const auto& data : localBuffer) {
            featFp << videoName << ";"
                   << encoderPreset << ";"
                   << targetQP << ";"
                   << bitDepth << ";"
                   << frameWidth << ";"
                   << frameHeight << ";"
                   << data.framePoc << ";"
                   << data.xPos << ";"
                   << data.yPos << ";"
                   << data.blockWidth << ";"
                   << data.blockHeight << ";" 
                   << data.blockArea << ";"
                   << data.blockAreaGroup << ";"
                   << data.frameLevel << ";"
                   << data.borderContactMask << ";"
                   << data.isIntra << ";"
                   << data.cuQp << ";"
                   << data.depth << ";"
                   << data.qtDepth << ";"
                   << data.btDepth << ";"
                   << data.splitSeries;

#if ENABLE_IMAGE_FEATURES_EXTRACTION == 1
            featFp << ";"
                   << data.blkPixelMean << ";"
                   << data.blkPixelVariance << ";"
                   << data.blkPixelStdDev << ";"
                   << data.blkPixelSum << ";"
                   << data.blkVarH << ";"
                   << data.blkVarV << ";"
                   << data.blkStdH << ";"
                   << data.blkStdV << ";"
                   << data.blkMin << ";"
                   << data.blkMax << ";"
                   << data.blkRange << ";"
                   << data.blkLaplacianVar << ";"
                   << data.blkEntropy << ";"
                   << data.blkSobelGv << ";"
                   << data.blkSobelGh << ";"
                   << data.blkSobelMag << ";"
                   << data.blkSobelDir << ";"
                   << data.blkSobelRazaoGrad << ";"
                   << data.blkPrewittGv << ";"
                   << data.blkPrewittGh << ";"
                   << data.blkPrewittMag << ";"
                   << data.blkPrewittDir << ";"
                   << data.blkPrewittRazaoGrad << ";"
                   << data.blkHadDc << ";"
                   << data.blkHadEnergyTotal << ";"
                   << data.blkHadEnergyAc << ";"
                   << data.blkHadMax << ";"
                   << data.blkHadMin << ";"
                   << data.blkHadTopLeft << ";"
                   << data.blkHadTopRight << ";"
                   << data.blkHadBottomLeft << ";"
                   << data.blkHadBottomRight; 
#endif
            featFp << "\n";
        }
    }
    localBuffer.clear();
}

void MLFeaturesManager::saveFeatures(const MLFeatureData& data) {
#if AVOID_FRAME_LEVEL_0 == 1
    if (data.frameLevel == 0)
        return;
#endif

    localBuffer.push_back(data);
    if (localBuffer.size() >= MAX_BUFFER_SIZE) {
        flushBuffer();
    }
}

void MLFeaturesManager::finish() {
    flushBuffer(); 
    if (featFp.is_open()) {
        featFp.close();
    }
}

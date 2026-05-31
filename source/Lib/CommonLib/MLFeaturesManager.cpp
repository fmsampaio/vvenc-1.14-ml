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

std::unordered_map<int, std::vector<MLFeatureData>> MLFeaturesManager::reservoirs;
std::unordered_map<int, uint64_t> MLFeaturesManager::seenCount;

thread_local std::vector<MLFeatureData> localBuffer;
thread_local std::mt19937 localRng(std::random_device{}());
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
               << "SplitSeries;"
               << "orientation_group;"
               << "aspect_ratio_group;"
               << "inter_cost;"
               << "cs_inter_had;"
               << "inter_had_per_pixel;"
               << "avail_left;"
               << "avail_above;"
               << "left_is_intra;"
               << "above_is_intra;"
               << "left_intra_dir;"
               << "above_intra_dir;"
               << "ref_line_variance;"
               << "ref_col_variance;"
               << "ref_line_mean;"
               << "ref_line_range;"
               << "ctu_pos_in_ctu_x;"
               << "ctu_pos_in_ctu_y;"
               << "is_first_line_of_ctu;"
               << "slice_type;"
               << "cu_mt_depth;"
               << "cu_bt_depth;"
               << "can_use_mip;"
               << "can_use_isp;"
               << "mpm0;"
               << "mpm_angular_var;"
               << "num_intra_ciip_neighbors;"
               << "relative_block_area;"
               << "delta_qp;"
               << "contrast_ratio;"
               << "directional_dominance;"
               << "variance_per_area;"
               << "mean_mismatch;"
               << "var_mismatch;"
               << "coef_variation;"
               << "ref_dominance;"
               << "mpm_delta;"
               << "dist_center_x;"
               << "dist_center_y;"
               << "boundary_complexity_ratio;"
               << "splitting_density;"
               << "center_focus_weight";
        
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
               << "blk_range";
               // Removed complex features
               // << ";" << "blk_laplacian_var"
               // << ";" << "blk_entropy"
               // << ";" << "blk_sobel_gv"
               // << ";" << "blk_sobel_gh"
               // << ";" << "blk_sobel_mag"
               // << ";" << "blk_sobel_dir"
               // << ";" << "blk_sobel_razao_grad"
               // << ";" << "blk_prewitt_gv"
               // << ";" << "blk_prewitt_gh"
               // << ";" << "blk_prewitt_mag"
               // << ";" << "blk_prewitt_dir"
               // << ";" << "blk_prewitt_razao_grad"
               // << ";" << "blk_had_dc"
               // << ";" << "blk_had_energy_total"
               // << ";" << "blk_had_energy_ac"
               // << ";" << "blk_had_max"
               // << ";" << "blk_had_min"
               // << ";" << "blk_had_topleft"
               // << ";" << "blk_had_topright"
               // << ";" << "blk_had_bottomleft"
               // << ";" << "blk_had_bottomright";
#endif
        featFp << "\n";
    } else {
        std::cerr << "[Error] MLFeaturesManager: Could not open file " << fileName << std::endl;
    }
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

void MLFeaturesManager::flushBuffer() {
    if (localBuffer.empty()) return;

    std::lock_guard<std::mutex> lock(writeMutex);

    for (const auto& data : localBuffer) {
        int isIntraVal = data.isIntra ? 1 : 0;
        int classKey = (data.frameLevel * 2) + isIntraVal; 

        uint64_t& count = seenCount[classKey];
        count++;

        auto& reservoir = reservoirs[classKey];

        if (reservoir.size() < RESERVOIR_SIZE) {
            reservoir.push_back(data);
        } else {
            std::uniform_int_distribution<uint64_t> dist(0, count - 1);
            uint64_t j = dist(localRng);

            if (j < RESERVOIR_SIZE) {
                reservoir[j] = data;
            }
        }
    }
    localBuffer.clear();
}

void MLFeaturesManager::finish() {
    flushBuffer();

    std::lock_guard<std::mutex> lock(writeMutex);

    if (!featFp.is_open())
        return;

    for (auto& entry : reservoirs) {
        auto& reservoir = entry.second;
        
        for (const auto& data : reservoir) {
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
                   << data.splitSeries << ";"
                   << data.orientationGroup << ";"
                   << data.aspectRatioGroup << ";"
                   << data.interCost << ";"
                   << data.csInterHad << ";"
                   << data.interHadPerPixel << ";"
                   << data.availLeft << ";"
                   << data.availAbove << ";"
                   << data.leftIsIntra << ";"
                   << data.aboveIsIntra << ";"
                   << data.leftIntraDir << ";"
                   << data.aboveIntraDir << ";"
                   << data.refLineVariance << ";"
                   << data.refColVariance << ";"
                   << data.refLineMean << ";"
                   << data.refLineRange << ";"
                   << data.ctuPosInCtuX << ";"
                   << data.ctuPosInCtuY << ";"
                   << data.isFirstLineOfCTU << ";"
                   << data.sliceType << ";"
                   << data.cuMtDepth << ";"
                   << data.cuBtDepth << ";"
                   << data.canUseMIP << ";"
                   << data.canUseISP << ";"
                   << data.mpm0 << ";"
                   << data.mpmAngularVar << ";"
                   << data.numIntraCiipNeighbors << ";"
                   << data.relativeBlockArea << ";"
                   << data.deltaQP << ";"
                   << data.contrastRatio << ";"
                   << data.directionalDominance << ";"
                   << data.variancePerArea << ";"
                   << data.meanMismatch << ";"
                   << data.varMismatch << ";"
                   << data.coefVariation << ";"
                   << data.refDominance << ";"
                   << data.mpmDelta << ";"
                   << data.distCenterX << ";"
                   << data.distCenterY << ";"
                   << data.boundaryComplexityRatio << ";"
                   << data.splittingDensity << ";"
                   << data.centerFocusWeight;

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
                   << data.blkRange;
                   // Removed complex features
                   // << ";" << data.blkLaplacianVar
                   // << ";" << data.blkEntropy
                   // << ";" << data.blkSobelGv
                   // << ";" << data.blkSobelGh
                   // << ";" << data.blkSobelMag
                   // << ";" << data.blkSobelDir
                   // << ";" << data.blkSobelRazaoGrad
                   // << ";" << data.blkPrewittGv
                   // << ";" << data.blkPrewittGh
                   // << ";" << data.blkPrewittMag
                   // << ";" << data.blkPrewittDir
                   // << ";" << data.blkPrewittRazaoGrad
                   // << ";" << data.blkHadDc
                   // << ";" << data.blkHadEnergyTotal
                   // << ";" << data.blkHadEnergyAc
                   // << ";" << data.blkHadMax
                   // << ";" << data.blkHadMin
                   // << ";" << data.blkHadTopLeft
                   // << ";" << data.blkHadTopRight
                   // << ";" << data.blkHadBottomLeft
                   // << ";" << data.blkHadBottomRight;
#endif
            featFp << "\n";
        }
    }

    featFp.close();
    
    reservoirs.clear();
    seenCount.clear();
}

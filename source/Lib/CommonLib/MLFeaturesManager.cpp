#include "MLFeaturesManager.h"
#include "CodingStructure.h"
#include "Unit.h"
#include "UnitTools.h"
#include "Picture.h"
#include "Reshape.h"

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
thread_local std::unordered_map<uint64_t, MLFeatureData> featureCache;
thread_local std::mt19937 localRng(std::random_device{}());
const size_t MAX_BUFFER_SIZE = 5000;

namespace
{
struct BufferStats
{
    double mean = 0.0;
    double variance = 0.0;
    double range = 0.0;
};

BufferStats calcBufferStats( const vvenc::CPelBuf& buf )
{
    BufferStats stats;

    if( buf.buf == nullptr || buf.width <= 0 || buf.height <= 0 )
    {
        return stats;
    }

    const int sampleCount = buf.width * buf.height;
    double sum = 0.0;
    double sumSq = 0.0;
    const vvenc::Pel* pBuf = buf.buf;
    double minVal = static_cast<double>( pBuf[0] );
    double maxVal = minVal;

    for( int y = 0; y < buf.height; y++ )
    {
        for( int x = 0; x < buf.width; x++ )
        {
            const double value = static_cast<double>( pBuf[x] );
            sum += value;
            sumSq += value * value;
            minVal = std::min( minVal, value );
            maxVal = std::max( maxVal, value );
        }
        pBuf += buf.stride;
    }

    stats.mean = sum / sampleCount;
    stats.variance = std::max( 0.0, ( sumSq / sampleCount ) - ( stats.mean * stats.mean ) );
    stats.range = maxVal - minVal;
    return stats;
}

void fillImageFeaturesFromBuf( const vvenc::CPelBuf& buf, MLFeatureData& data )
{
    if( buf.buf == nullptr || buf.width <= 0 || buf.height <= 0 )
    {
        return;
    }

    const int sampleCount = buf.width * buf.height;
    std::vector<double> colSum( buf.width, 0.0 );
    std::vector<double> colSumSq( buf.width, 0.0 );

    const vvenc::Pel* pBuf = buf.buf;
    double sum = 0.0;
    double sumSq = 0.0;
    double rowVarSum = 0.0;
    double colVarSum = 0.0;
    double rowStdSum = 0.0;
    double colStdSum = 0.0;
    double minVal = static_cast<double>( pBuf[0] );
    double maxVal = minVal;

    for( int y = 0; y < buf.height; y++ )
    {
        double rowSum = 0.0;
        double rowSumSq = 0.0;

        for( int x = 0; x < buf.width; x++ )
        {
            const double value = static_cast<double>( pBuf[x] );
            sum += value;
            sumSq += value * value;
            rowSum += value;
            rowSumSq += value * value;
            colSum[x] += value;
            colSumSq[x] += value * value;
            minVal = std::min( minVal, value );
            maxVal = std::max( maxVal, value );
        }

        const double rowMean = rowSum / buf.width;
        const double rowVar = std::max( 0.0, ( rowSumSq / buf.width ) - ( rowMean * rowMean ) );
        rowVarSum += rowVar;
        rowStdSum += std::sqrt( rowVar );

        pBuf += buf.stride;
    }

    for( int x = 0; x < buf.width; x++ )
    {
        const double colMean = colSum[x] / buf.height;
        const double colVar = std::max( 0.0, ( colSumSq[x] / buf.height ) - ( colMean * colMean ) );
        colVarSum += colVar;
        colStdSum += std::sqrt( colVar );
    }

    data.blkPixelMean = sum / sampleCount;
    data.blkPixelVariance = std::max( 0.0, ( sumSq / sampleCount ) - ( data.blkPixelMean * data.blkPixelMean ) );
    data.blkPixelStdDev = std::sqrt( data.blkPixelVariance );
    data.blkPixelSum = sum;
    data.blkVarH = rowVarSum / buf.height;
    data.blkVarV = colVarSum / buf.width;
    data.blkStdH = rowStdSum / buf.height;
    data.blkStdV = colStdSum / buf.width;
    data.blkMin = minVal;
    data.blkMax = maxVal;
    data.blkRange = maxVal - minVal;

    if( buf.width == buf.height )
    {
        data.orientationGroup = 0;
    }
    else if( buf.width > buf.height )
    {
        data.orientationGroup = 1;
    }
    else
    {
        data.orientationGroup = 2;
    }

    const int minDim = std::min( buf.width, buf.height );
    const int maxDim = std::max( buf.width, buf.height );
    const int ratio = maxDim / std::max( 1, minDim );
    data.aspectRatioGroup = ratio > 0 ? static_cast<int>( std::log2( ratio ) + 0.5 ) : 0;

    const int fw = MLFeaturesManager::getFrameWidth();
    const int fh = MLFeaturesManager::getFrameHeight();
    if( fw > 0 && fh > 0 )
    {
        data.relativeBlockArea = (double) data.blockArea / (double) ( fw * fh );
        const double centerX = fw / 2.0;
        const double centerY = fh / 2.0;
        data.distCenterX = std::abs( data.xPos + ( buf.width / 2.0 ) - centerX ) / centerX;
        data.distCenterY = std::abs( data.yPos + ( buf.height / 2.0 ) - centerY ) / centerY;
        const double distSq = ( data.distCenterX * data.distCenterX ) + ( data.distCenterY * data.distCenterY );
        data.centerFocusWeight = 1.0 / ( std::sqrt( distSq ) + 1.0 );
    }
    else
    {
        data.relativeBlockArea = 0.0;
        data.distCenterX = 0.0;
        data.distCenterY = 0.0;
        data.centerFocusWeight = 0.0;
    }

    data.deltaQP = data.cuQp - MLFeaturesManager::getTargetQP();
    data.contrastRatio = data.blkRange / ( data.blkPixelMean + 1.0 );
    data.directionalDominance = std::abs( data.blkVarH - data.blkVarV ) / ( data.blkVarH + data.blkVarV + 1.0 );
    data.variancePerArea = data.blockArea > 0 ? data.blkPixelVariance / (double) data.blockArea : 0.0;
    data.meanMismatch = std::abs( data.blkPixelMean - data.refLineMean );
    data.varMismatch = std::abs( data.blkPixelVariance - data.refLineVariance );
    data.coefVariation = data.blkPixelStdDev / ( data.blkPixelMean + 1.0 );
    data.refDominance = data.refLineVariance / ( data.refColVariance + 1.0 );
    data.mpmDelta = ( data.leftIsIntra && data.aboveIsIntra ) ? std::abs( data.leftIntraDir - data.aboveIntraDir ) : -1;
    data.boundaryComplexityRatio = ( data.refLineVariance + data.refColVariance ) / ( data.blkPixelVariance + 1.0 );
    data.splittingDensity = ( data.cuMtDepth + data.cuBtDepth ) / 6.0;
}
}

MLFeatureData MLFeaturesManager::extractFeatures(const vvenc::CodingStructure& cs, const vvenc::CodingUnit& cu, double bestCostInter)
{
    MLFeatureData featData;

    featData.framePoc    = cu.slice->poc;
    featData.frameLevel  = cu.slice->TLayer;
    featData.xPos        = cu.lx();
    featData.yPos        = cu.ly();
    featData.blockWidth  = cu.lwidth();
    featData.blockHeight = cu.lheight();
    featData.depth       = cu.depth;
    featData.qtDepth     = cu.qtDepth;
    featData.btDepth     = cu.btDepth;
    featData.cuMtDepth   = cu.mtDepth;
    featData.cuBtDepth   = cu.btDepth;
    featData.splitSeries = (long long) cu.splitSeries;
    featData.cuQp        = cu.qp;

    featData.interCost = ( bestCostInter >= 1e300 ) ? -1.0 : bestCostInter;
    featData.csInterHad = ( cu.cs->interHad >= 1e18 ) ? -1.0 : static_cast<double>( cu.cs->interHad );

    const unsigned ctuSize = cu.slice->pps ? cu.slice->pps->ctuSize : 0;
    featData.ctuPosInCtuX = ctuSize > 0 ? featData.xPos % (int) ctuSize : 0;
    featData.ctuPosInCtuY = ctuSize > 0 ? featData.yPos % (int) ctuSize : 0;
    featData.isFirstLineOfCTU = ( featData.ctuPosInCtuY == 0 );
    featData.sliceType = cu.slice->isIntra() ? 0 : ( cu.slice->isInterP() ? 1 : 2 );

    featData.blockArea = featData.blockWidth * featData.blockHeight;
    featData.blockAreaGroup = featData.blockArea > 0 ? (int) std::log2( (double) featData.blockArea ) - 4 : 0;
    featData.interHadPerPixel = featData.blockArea > 0 ? featData.csInterHad / (double) featData.blockArea : 0.0;

    featData.borderContactMask = 0;
    if( featData.xPos == 0 ) featData.borderContactMask += 1;
    if( ( featData.xPos + featData.blockWidth ) == frameWidth ) featData.borderContactMask += 2;
    if( featData.yPos == 0 ) featData.borderContactMask += 4;
    if( ( featData.yPos + featData.blockHeight ) == frameHeight ) featData.borderContactMask += 8;

    featData.canUseMIP = cu.slice->sps->MIP && cu.lwidth() <= cu.slice->sps->getMaxTbSize() && cu.lheight() <= cu.slice->sps->getMaxTbSize();
    featData.canUseISP = cu.slice->sps->ISP && vvenc::CU::canUseISP( cu.lwidth(), cu.lheight(), cu.slice->sps->getMaxTbSize() );

    unsigned mpmList[vvenc::NUM_LUMA_MODE] = { 0 };
    const int numMPMs = vvenc::CU::getIntraMPMs( cu, mpmList );
    int angularMPMs[vvenc::NUM_LUMA_MODE] = { 0 };
    int numAngularMPMs = 0;
    for( int i = 0; i < numMPMs; i++ )
    {
        if( (int) mpmList[i] > vvenc::DC_IDX )
        {
            angularMPMs[numAngularMPMs++] = (int) mpmList[i];
        }
    }
    featData.mpm0 = numAngularMPMs > 0 ? angularMPMs[0] : ( numMPMs > 0 ? (int) mpmList[0] : -1 );
    if( numAngularMPMs > 0 )
    {
        double mpmMean = 0.0;
        for( int i = 0; i < numAngularMPMs; i++ )
        {
            mpmMean += angularMPMs[i];
        }
        mpmMean /= numAngularMPMs;
        double mpmVar = 0.0;
        for( int i = 0; i < numAngularMPMs; i++ )
        {
            const double diff = angularMPMs[i] - mpmMean;
            mpmVar += diff * diff;
        }
        featData.mpmAngularVar = mpmVar / numAngularMPMs;
    }

    const vvenc::CodingUnit* leftCu = cs.getCURestricted( cu.lumaPos().offset( -1, 0 ), cu, vvenc::CH_L );
    const vvenc::CodingUnit* aboveCu = cs.getCURestricted( cu.lumaPos().offset( 0, -1 ), cu, vvenc::CH_L );
    const vvenc::CodingUnit* aboveLeftCu = cs.getCURestricted( cu.lumaPos().offset( -1, -1 ), cu, vvenc::CH_L );
    const vvenc::CodingUnit* aboveRightCu = cs.getCURestricted( cu.lumaPos().offset( cu.lwidth(), -1 ), cu, vvenc::CH_L );

    featData.availLeft = leftCu != nullptr;
    featData.availAbove = aboveCu != nullptr;
    featData.leftIsIntra = leftCu != nullptr && leftCu->predMode == vvenc::PredMode::MODE_INTRA;
    featData.aboveIsIntra = aboveCu != nullptr && aboveCu->predMode == vvenc::PredMode::MODE_INTRA;
    featData.leftIntraDir = leftCu != nullptr ? leftCu->intraDir[vvenc::CH_L] : -1;
    featData.aboveIntraDir = aboveCu != nullptr ? aboveCu->intraDir[vvenc::CH_L] : -1;
    featData.numIntraCiipNeighbors = ( featData.leftIsIntra ? 1 : 0 ) + ( featData.aboveIsIntra ? 1 : 0 );

    featData.leftDepth = leftCu ? leftCu->depth : -1;
    featData.aboveDepth = aboveCu ? aboveCu->depth : -1;
    featData.leftQtDepth = leftCu ? leftCu->qtDepth : -1;
    featData.aboveQtDepth = aboveCu ? aboveCu->qtDepth : -1;
    featData.leftMtDepth = leftCu ? leftCu->mtDepth : -1;
    featData.aboveMtDepth = aboveCu ? aboveCu->mtDepth : -1;

    featData.numIntraNeighbors4 = 0;
    if( featData.leftIsIntra ) featData.numIntraNeighbors4++;
    if( featData.aboveIsIntra ) featData.numIntraNeighbors4++;
    if( aboveLeftCu && aboveLeftCu->predMode == vvenc::PredMode::MODE_INTRA ) featData.numIntraNeighbors4++;
    if( aboveRightCu && aboveRightCu->predMode == vvenc::PredMode::MODE_INTRA ) featData.numIntraNeighbors4++;

    if( featData.yPos > 0 )
    {
        vvenc::CompArea aboveArea = cu.Y();
        aboveArea.y -= 1;
        aboveArea.height = 1;
        const BufferStats aboveStats = calcBufferStats( cs.picture->getOrigBuf( aboveArea ) );
        featData.refLineMean = aboveStats.mean;
        featData.refLineVariance = aboveStats.variance;
        featData.refLineRange = aboveStats.range;
    }
    if( featData.xPos > 0 )
    {
        vvenc::CompArea leftArea = cu.Y();
        leftArea.x -= 1;
        leftArea.width = 1;
        const BufferStats leftStats = calcBufferStats( cs.picture->getOrigBuf( leftArea ) );
        featData.refColVariance = leftStats.variance;
    }

    const auto& reshapeData = cs.picture->reshapeData;
    const vvenc::CPelBuf orgBuf = ( cs.picHeader->lmcsEnabled && reshapeData.getCTUFlag() )
        ? cs.getRspOrgBuf( cu.Y() )
        : cs.getOrgBuf( cu.Y() );

    fillImageFeaturesFromBuf( orgBuf, featData );

    return featData;
}

uint64_t MLFeaturesManager::getCuKey(int poc, int x, int y, int w, int h, int qtDepth, int mtDepth)
{
    uint64_t key = 1469598103934665603ull;

    auto mix = [&key](uint64_t value)
    {
        key ^= value;
        key *= 1099511628211ull;
    };

    mix( static_cast<uint64_t>( static_cast<uint32_t>( poc ) ) );
    mix( static_cast<uint64_t>( static_cast<uint32_t>( x ) ) );
    mix( static_cast<uint64_t>( static_cast<uint32_t>( y ) ) );
    mix( static_cast<uint64_t>( static_cast<uint32_t>( w ) ) );
    mix( static_cast<uint64_t>( static_cast<uint32_t>( h ) ) );

    return key;
}

void MLFeaturesManager::cacheFeatures(uint64_t key, const MLFeatureData& data)
{
    featureCache[key] = data;
}

MLFeatureData* MLFeaturesManager::getCachedFeatures(uint64_t key)
{
    auto it = featureCache.find(key);
    return it != featureCache.end() ? &it->second : nullptr;
}

void MLFeaturesManager::eraseCachedFeatures(uint64_t key)
{
    featureCache.erase(key);
}

void MLFeaturesManager::clearCache()
{
    featureCache.clear();
}

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
               << "left_depth;"
               << "above_depth;"
               << "left_qt_depth;"
               << "above_qt_depth;"
               << "left_mt_depth;"
               << "above_mt_depth;"
               << "num_intra_neighbors4;"
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
                   << data.leftDepth << ";"
                   << data.aboveDepth << ";"
                   << data.leftQtDepth << ";"
                   << data.aboveQtDepth << ";"
                   << data.leftMtDepth << ";"
                   << data.aboveMtDepth << ";"
                   << data.numIntraNeighbors4 << ";"
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
            featFp << "\n";
        }
    }

    featFp.close();
    
    reservoirs.clear();
    seenCount.clear();
}

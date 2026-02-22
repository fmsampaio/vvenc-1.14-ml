#include "MLFeaturesManager.h"

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
        featFp << "VideoName;EncoderPreset;TargetQP;BitDepth;FrameWidth;FrameHeight;Frame;X_Pos;Y_Pos;BlockWidth;BlockHeight;BlockArea;BlockAreaGroup;FrameLevel;BorderContactMask;IsIntra;CU_QP;Depth;QTDepth;BTDepth;SplitSeries\n";
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
                   << data.splitSeries << "\n";
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

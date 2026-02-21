#include "MLFeaturesManager.h"

std::ofstream MLFeaturesManager::featFp;
std::mutex MLFeaturesManager::writeMutex;

thread_local std::vector<MLFeatureData> localBuffer;

const size_t MAX_BUFFER_SIZE = 5000; 

void MLFeaturesManager::init(const std::string& fileName) {
    featFp.open(fileName);
    if (featFp.is_open()) {
        featFp << "frame_width;frame_height;frame_poc;frame_level;x_pos;y_pos;block_width;block_height;is_intra\n";
    } else {
        std::cerr << "[Error] MLFeaturesManager: Could not open file " << fileName << std::endl;
    }
}

void MLFeaturesManager::flushBuffer() {
    if (localBuffer.empty()) return;

    std::lock_guard<std::mutex> lock(writeMutex);
    
    if (featFp.is_open()) {
        for (const auto& data : localBuffer) {
            featFp << data.frameWidth << ";"
                   << data.frameHeight << ";"
                   << data.framePoc << ";"
                   << data.frameLevel << ";"
                   << data.xPos << ";"
                   << data.yPos << ";"
                   << data.blockWidth << ";"
                   << data.blockHeight << ";" 
                   << data.isIntra << "\n";
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

#include "MLFeaturesManager.h"

std::ofstream MLFeaturesManager::featFp;
int MLFeaturesManager::xPos, MLFeaturesManager::yPos, MLFeaturesManager::blockWidth, MLFeaturesManager::blockHeight;
int MLFeaturesManager::frameWidth, MLFeaturesManager::frameHeight, MLFeaturesManager::framePoc, MLFeaturesManager::frameLevel;
bool MLFeaturesManager::isIntra;

void MLFeaturesManager::init() {
    featFp.open("/home/felipe/Projetos/vvenc-1.14-ml/bin/release-static/features.csv");
    featFp << "frame_width;frame_height;frame_poc;frame_level;x_pos;y_pos;block_width;block_height;is_intra" << std::endl;
}

void MLFeaturesManager::finish() {
    featFp.close();
}

void MLFeaturesManager::addFeaturesLine() {

#if AVOID_FRAME_LEVEL_0
    if(frameLevel == 0) 
        return;
#endif

    featFp << frameWidth << ";";
    featFp << frameHeight << ";";
    featFp << framePoc << ";";
    featFp << frameLevel << ";";
    
    featFp << xPos << ";";
    featFp << yPos << ";";
    featFp << blockWidth << ";";
    featFp << blockWidth << ";";

    featFp << isIntra;
    featFp << std::endl;
}

void MLFeaturesManager::collectFrameParameters(int frameWidth, int frameHeight, int framePoc, int frameLevel) {
    MLFeaturesManager::frameWidth = frameWidth;
    MLFeaturesManager::frameHeight = frameHeight;
    MLFeaturesManager::framePoc = framePoc;
    MLFeaturesManager::frameLevel = frameLevel;
}

void MLFeaturesManager::collectBlockParameters(int xPos, int yPos, int blockWidth, int blockHeight) {
    MLFeaturesManager::xPos = xPos;
    MLFeaturesManager::yPos = yPos;
    MLFeaturesManager::blockWidth = blockWidth;
    MLFeaturesManager::blockHeight = blockHeight;
}

void MLFeaturesManager::collectPredMode(vvenc::PredMode predMode) {
    isIntra = (predMode == vvenc::PredMode::MODE_INTRA);
}
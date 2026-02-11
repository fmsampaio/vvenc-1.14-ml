#include "MLFeaturesManager.h"

std::ofstream MLFeaturesManager::featFp;
int MLFeaturesManager::xPos, MLFeaturesManager::yPos, MLFeaturesManager::blockWidth, MLFeaturesManager::blockHeight;

void MLFeaturesManager::init() {
    featFp.open("/home/research-data-ssd-1/felipe/vvenc-1.14-ml/bin/release-static/features.csv");
    featFp << "dummy_feat_1;dummy_feat_2" << std::endl;
}

void MLFeaturesManager::finish() {
    featFp.close();
}

void MLFeaturesManager::addFeaturesLine() {
    featFp << ";\n";
}

void MLFeaturesManager::collectBlockParameters(int xPos, int yPos, int blockWidth, int blockHeight) {
    MLFeaturesManager::xPos = xPos;
    MLFeaturesManager::yPos = yPos;
    MLFeaturesManager::blockWidth = blockWidth;
    MLFeaturesManager::blockHeight = blockHeight;
}

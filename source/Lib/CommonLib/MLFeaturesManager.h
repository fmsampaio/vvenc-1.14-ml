#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <vector>
#include "CommonDef.h"

#define ENABLE_FEATURES_EXTRACTION 1
#define AVOID_FRAME_LEVEL_0 1

struct MLFeatureData {
    int frameWidth;
    int frameHeight;
    int framePoc;
    int frameLevel;
    int xPos;
    int yPos;
    int blockWidth;
    int blockHeight;
    bool isIntra;
};

class VVENC_DECL MLFeaturesManager {
private:
    static std::ofstream featFp;
    static std::mutex writeMutex;

public:
    static void init(const std::string& fileName);
    static void finish();
    static void saveFeatures(const MLFeatureData& data);
    
    static void flushBuffer(); 
};
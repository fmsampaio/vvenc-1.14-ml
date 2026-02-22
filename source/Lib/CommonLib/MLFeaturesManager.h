#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <vector>
#include <cmath>
#include "CommonDef.h"

#define ENABLE_FEATURES_EXTRACTION 1
#define AVOID_FRAME_LEVEL_0 1

struct MLFeatureData {
    int framePoc;
    int frameLevel;
    int xPos;
    int yPos;
    int blockWidth;
    int blockHeight;
    int blockArea;
    int blockAreaGroup;
    int borderContactMask;
    bool isIntra;
    int cuQp;
    int depth;
    int qtDepth;
    int btDepth;
    long long splitSeries;
};

class VVENC_DECL MLFeaturesManager {
private:
    static std::ofstream featFp;
    static std::mutex writeMutex;

    static std::string videoName;
    static std::string encoderPreset;
    static int targetQP;
    static int bitDepth;
    static int frameWidth;
    static int frameHeight;

public:
    static void init(const std::string& fileName, const std::string& vName, 
                     const std::string& preset, int tQp, int bDepth, int fWidth, int fHeight);
    static void finish();
    static void saveFeatures(const MLFeatureData& data);
    static void flushBuffer(); 

    static int getFrameWidth() { return frameWidth; }
    static int getFrameHeight() { return frameHeight; }
};
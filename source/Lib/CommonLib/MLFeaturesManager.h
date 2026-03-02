#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <random>
#include "CommonDef.h"

#define ENABLE_FEATURES_EXTRACTION 0
#define AVOID_FRAME_LEVEL_0 1

#if ENABLE_FEATURES_EXTRACTION
    #define ENABLE_IMAGE_FEATURES_EXTRACTION 1
#endif

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

#if ENABLE_IMAGE_FEATURES_EXTRACTION
    double blkPixelMean, blkPixelVariance, blkPixelStdDev, blkPixelSum;
    double blkVarH, blkVarV, blkStdH, blkStdV;
    double blkMin, blkMax, blkRange;
    double blkLaplacianVar, blkEntropy;
    double blkSobelGv, blkSobelGh, blkSobelMag, blkSobelDir, blkSobelRazaoGrad;
    double blkPrewittGv, blkPrewittGh, blkPrewittMag, blkPrewittDir, blkPrewittRazaoGrad;
    double blkHadDc, blkHadEnergyTotal, blkHadEnergyAc, blkHadMax, blkHadMin;
    double blkHadTopLeft, blkHadTopRight, blkHadBottomLeft, blkHadBottomRight;
#endif
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

    static constexpr size_t RESERVOIR_SIZE = 200000; // 200k samples per (frameLevel × intra/inter) class
    static std::unordered_map<int, std::vector<MLFeatureData>> reservoirs; // key = classId (frameLevel × intra/inter)
    static std::unordered_map<int, uint64_t> seenCount;

public:
    static void init(const std::string& fileName, const std::string& vName, 
                     const std::string& preset, int tQp, int bDepth, int fWidth, int fHeight);
    static void finish();
    static void saveFeatures(const MLFeatureData& data);
    static void flushBuffer(); 

    static int getFrameWidth() { return frameWidth; }
    static int getFrameHeight() { return frameHeight; }
};
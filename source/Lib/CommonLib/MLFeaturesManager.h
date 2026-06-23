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
#include <atomic>

namespace vvenc {
    class CodingStructure;
    class CodingUnit;
}

#define ENABLE_FEATURES_EXTRACTION 1
#define AVOID_FRAME_LEVEL_0 1
#define USE_RESERVOIR 1                 // 0 = turns off reservoir (records everything)

struct MLFeatureData {
    // --- Global and position information ---
    int framePoc;                   // Picture Order Count (display order of the frame)
    int frameLevel;                 // Hierarchical frame level in Random Access
    int xPos;                       // X position (horizontal) of the block in pixels
    int yPos;                       // Y position (vertical) of the block in pixels
    int blockWidth;                 // Block width in pixels
    int blockHeight;                // Block height in pixels
    int blockArea;                  // Total area of the block (width * height)
    int blockAreaGroup;             // Block area group (log2 of the largest dimension)
    int borderContactMask;          // Mask indicating whether the block touches a frame border
    
    // --- Target labels ---
    bool isIntra;                   // Split-aware target: the best NON-split mode for this block was intra
                                    // (true even if the block was later split). "O split foi intra ou nao?"
    int finalDecision;              // Final decision for the partition-tree node. When the node is kept as a
                                    // single CU spanning the full area, holds the CU prediction mode
                                    // (0=INTER, 1=INTRA, 2=IBC, 3=PLT); 4=SPLIT when the block is divided.
                                    // Derive IntraKept = (finalDecision == 1); wasSplit = (finalDecision == 4).
    bool isSplit;                   // True if the block was split (finalDecision == 4)
    
    // --- CU coding information ---
    int cuQp;                       // Local Quantization Parameter (QP) of the CU
    int depth;                      // Overall partition tree depth
    int qtDepth;                    // Quad-tree (QT) depth
    int btDepth;                    // Binary-tree (BT) depth
    long long splitSeries;          // Representative history of splits applied to the block

    // --- Block geometry ---
    int orientationGroup;           // Block orientation (0=square, 1=horizontal, 2=vertical)
    int aspectRatioGroup;           // Aspect ratio group (log2 of the ratio between dimensions)
    
    // --- Inter prediction cost ---
    double interCost;               // Estimated inter prediction cost
    double csInterHad;              // Best inter SATD cost (SATD/Hadamard)
    double interHadPerPixel;        // Inter SATD cost normalized by block area
    
    // --- Neighbor availability and modes ---
    bool availLeft;                 // Whether a reconstructed left neighbor exists
    bool availAbove;                // Whether a reconstructed above neighbor exists
    bool leftIsIntra;               // Whether the left neighbor used intra coding
    bool aboveIsIntra;              // Whether the above neighbor used intra coding
    int leftIntraDir;               // Intra direction angle used by the left neighbor
    int aboveIntraDir;              // Intra direction angle used by the above neighbor
    
    // --- Reference line statistics ---
    double refLineVariance;         // Variance of pixels in the above reference line
    double refColVariance;          // Variance of pixels in the left reference column
    double refLineMean;             // Mean of pixels in the above reference line
    double refLineRange;            // Range (max - min) of pixels in the above reference line
    
    // --- CTU and slice information ---
    int ctuPosInCtuX;               // Horizontal position inside the current CTU
    int ctuPosInCtuY;               // Vertical position inside the current CTU
    bool isFirstLineOfCTU;          // True if the block is in the first CTU line
    int sliceType;                  // Slice type (0=I, 1=P, 2=B)
    
    // --- Extended depth and intra tools ---
    int cuMtDepth;                  // Depth reached in the multi-type tree
    int cuBtDepth;                  // Number of consecutive half-splits applied
    bool canUseMIP;                 // Whether the block meets MIP constraints
    bool canUseISP;                 // Whether the block meets ISP constraints
    
    // --- Intra prediction and MPM ---
    int mpm0;                       // Most probable mode (MPM0) suggested by neighbors
    double mpmAngularVar;           // Angular disagreement between neighbor suggestions
    int numIntraCiipNeighbors;      // Number of intra neighbors used for CIIP
    
    // --- Neighbor depths and extended context ---
    int leftDepth;                  // Depth of left neighbor CU
    int aboveDepth;                 // Depth of above neighbor CU
    int leftQtDepth;                // QT depth of left neighbor CU
    int aboveQtDepth;               // QT depth of above neighbor CU
    int leftMtDepth;                // MT depth of left neighbor CU
    int aboveMtDepth;               // MT depth of above neighbor CU
    int numIntraNeighbors4;         // Number of intra neighbors among left/above/above-left/above-right

    // --- Spatial and temporal context ---
    bool leftIsMIP;                 // Whether left neighbor uses MIP
    bool aboveIsMIP;                // Whether above neighbor uses MIP
    int leftISPMode;                // ISP mode used by left neighbor
    int aboveISPMode;               // ISP mode used by above neighbor
    double neighborMeanDepth;       // Mean depth among available 4-neighborhood

    // --- Temporal context (collocated region, 4x4 motion grid)
    int refIsIntra;                 // 1 if any collocated 4x4 in the ref region is non-inter (intra/IBC)
    double refIntraAreaRatio;       // Ratio of non-inter (intra/IBC) 4x4 areas in collocated ref region
    double refBiPredAreaRatio;      // Ratio of bi-predicted 4x4 areas (interDir==3) in collocated ref region
    double refMvAvgMagnitude;       // Mean L0 motion magnitude (|hor|+|ver|) over inter 4x4 areas
    double refMvMagnitudeVar;       // Variance of L0 motion magnitude over inter 4x4 areas (motion heterogeneity)

    // --- Safe derived features ---
    double relativeBlockArea;       // Block area proportion relative to the full frame
    int deltaQP;                    // Difference between block QP and encoder target QP
    double contrastRatio;           // Pixel amplitude to mean ratio (range / mean)
    double directionalDominance;    // Normalized difference between horizontal and vertical variance
    double variancePerArea;         // Block variance divided by its area
    double meanMismatch;            // Absolute difference between block mean and reference mean
    double varMismatch;             // Absolute difference between block variance and reference variance
    double coefVariation;           // Standard deviation divided by mean
    double refDominance;            // Complexity ratio between above and left references
    int mpmDelta;                   // Absolute angular distance between above and left neighbors
    double distCenterX;             // Normalized horizontal distance to the frame center
    double distCenterY;             // Normalized vertical distance to the frame center
    double boundaryComplexityRatio; // Sum of neighbor variances divided by block variance
    double splittingDensity;        // Sum of tree depths divided by a generic maximum depth
    double centerFocusWeight;       // Weight inversely proportional to distance from the center

    // --- Original image ---
    double blkPixelMean;            // Mean pixel value of the original block
    double blkPixelVariance;        // Statistical variance of block pixels
    double blkPixelStdDev;          // Standard deviation of block pixels
    double blkPixelSum;             // Sum of all pixel values in the block
    double blkVarH;                 // Row-wise variance projection (horizontal)
    double blkVarV;                 // Column-wise variance projection (vertical)
    double blkStdH;                 // Row-wise standard deviation (horizontal)
    double blkStdV;                 // Column-wise standard deviation (vertical)
    double blkMin;                  // Minimum pixel value inside the block
    double blkMax;                  // Maximum pixel value inside the block
    double blkRange;                // Difference between max and min pixel values

    // Constructor for safe initialization
    MLFeatureData() :
        isIntra(false), finalDecision(-1), isSplit(false),
        cuQp(0), depth(0), qtDepth(0), btDepth(0), splitSeries(0),
        orientationGroup(0), aspectRatioGroup(0),
        interCost(-1.0), csInterHad(0.0), interHadPerPixel(-1.0),
        availLeft(false), availAbove(false), leftIsIntra(false), aboveIsIntra(false),
        leftIntraDir(-1), aboveIntraDir(-1),
        refLineVariance(0.0), refColVariance(0.0), refLineMean(0.0), refLineRange(0.0),
        ctuPosInCtuX(0), ctuPosInCtuY(0), isFirstLineOfCTU(false), sliceType(0),
        cuMtDepth(0), cuBtDepth(0), canUseMIP(false), canUseISP(false),
        mpm0(-1), mpmAngularVar(0.0), numIntraCiipNeighbors(0),
        leftDepth(-1), aboveDepth(-1), leftQtDepth(-1), aboveQtDepth(-1),
        leftMtDepth(-1), aboveMtDepth(-1), numIntraNeighbors4(0),
        leftIsMIP(false), aboveIsMIP(false), leftISPMode(0), aboveISPMode(0),
        neighborMeanDepth(-1.0), refIsIntra(0), refIntraAreaRatio(0.0),
        refBiPredAreaRatio(0.0), refMvAvgMagnitude(0.0), refMvMagnitudeVar(0.0),
        relativeBlockArea(0.0), deltaQP(0), contrastRatio(0.0), directionalDominance(0.0), variancePerArea(0.0),
        meanMismatch(0.0), varMismatch(0.0), coefVariation(0.0), refDominance(0.0), mpmDelta(-1),
        distCenterX(0.0), distCenterY(0.0), boundaryComplexityRatio(0.0), 
        splittingDensity(0.0), centerFocusWeight(0.0),
        blkPixelMean(0.0), blkPixelVariance(0.0), blkPixelStdDev(0.0), blkPixelSum(0.0),
        blkVarH(0.0), blkVarV(0.0), blkStdH(0.0), blkStdV(0.0),
        blkMin(0.0), blkMax(0.0), blkRange(0.0) {}
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

    static constexpr size_t RESERVOIR_SIZE = 5000; // samples per (frameLevel × intra/inter) class
    static std::unordered_map<int, std::vector<MLFeatureData>> reservoirs; // key = classId (frameLevel × intra/inter)
    static std::unordered_map<int, uint64_t> seenCount;

    static void writeRow(const MLFeatureData& data);

public:
    static void init(const std::string& fileName, const std::string& vName, 
                     const std::string& preset, int tQp, int bDepth, int fWidth, int fHeight);
    static void finish();
    static MLFeatureData extractFeatures(const vvenc::CodingStructure& cs, const vvenc::CodingUnit& cu, double bestCostInter);
    static void saveFeatures(const MLFeatureData& data);
    static uint64_t getCuKey(int poc, int x, int y, int w, int h, int qtDepth, int mtDepth);
    static void cacheFeatures(uint64_t key, const MLFeatureData& data);
    static MLFeatureData* getCachedFeatures(uint64_t key);
    static void eraseCachedFeatures(uint64_t key);
    static void clearCache();

    static int getFrameWidth() { return frameWidth; }
    static int getFrameHeight() { return frameHeight; }
    static int getTargetQP() { return targetQP; }

    static std::atomic<uint64_t> globalTotalBlocks;
    static std::atomic<uint64_t> intraLumaEvaluated;
    static std::atomic<uint64_t> intraChromaEvaluated;
    static std::atomic<uint64_t> intraTotalEvaluated;
    static std::atomic<uint64_t> intraBlocksKept;
    static std::atomic<uint64_t> blocksSplit;
    static std::atomic<uint64_t> blocksOther;
    static std::atomic<uint64_t> datasetSavedRows;
    static std::atomic<uint64_t> intraLumaFrameLevel0;

    static void printOptimizationPotential();
};
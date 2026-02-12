#include <iostream>
#include <fstream>

#include "CommonDef.h"


#define ENABLE_FEATURES_EXTRACTION 0
#define AVOID_FRAME_LEVEL_0 1

class MLFeaturesManager {
    private:
        static std::ofstream featFp;

        // Features list
        static int xPos, yPos, blockWidth, blockHeight;
        static int frameWidth, frameHeight, framePoc, frameLevel;
        static bool isIntra;

    public:
        static void init(std::string fileName);
        static void finish();

        static void addFeaturesLine();

        static void collectFrameParameters(int frameWidth, int frameHeight, int framePoc, int frameLevel);
        static void collectBlockParameters(int xPos, int yPos, int blockWidth, int blockHeight);
        static void collectPredMode(vvenc::PredMode predMode);

};
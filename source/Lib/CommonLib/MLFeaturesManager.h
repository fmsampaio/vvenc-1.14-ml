#include <iostream>
#include <fstream>

#include "TypeDef.h"


#define ENABLE_FEATURES_EXTRACTION 1

class MLFeaturesManager {
    private:
        static std::ofstream featFp;

        // Features list
        static int xPos, yPos, blockWidth, blockHeight;

        static vvenc::PredMode predMode;

    public:
        static void init();
        static void finish();

        static void addFeaturesLine();

        static void collectBlockParameters(int xPos, int yPos, int blockWidth, int blockHeight);

};
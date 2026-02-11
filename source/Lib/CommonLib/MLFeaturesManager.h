#include <iostream>
#include <fstream>

#define ENABLE_FEATURES_EXTRACTION 1

class MLFeaturesManager {
    private:
        static std::ofstream featFp;

    public:
        static void init();
        static void finish();

        static void addFeaturesLine();

};
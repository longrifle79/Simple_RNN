#include <random>

class RandomizeWeight
{
    private:
        float maximum;
        float minimum;
    public:
        RandomizeWeight();
        RandomizeWeight(float min, float max);

        static float getFloat(float min, float max);
        float getFloat();
};
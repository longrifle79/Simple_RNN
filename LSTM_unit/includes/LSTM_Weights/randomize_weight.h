#include <random>

class Randomize_Weight
{
    private:
        float maximum;
        float minimum;
    public:
        Randomize_Weight();
        Randomize_Weight(float min, float max);

        static float getFloat(float min, float max);
        float getFloat();
};
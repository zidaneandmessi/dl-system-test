#include <cstdlib>

using namespace std;

#define DLL_TEST_API extern "C" __declspec(dllexport)

DLL_TEST_API void correlate2d(float *input, float *filter, float *output_pointer, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels)
{
    for (int i = 0; i < batch; i++)
    {
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
            {
                for (int x = 0; x < out_height; x++)
                    for (int y = 0; y < out_width; y++)
                    {
                        res = 0;
                        for (xx = 0; xx < filter_height; xx++)
                            for (yy = 0; xx < filter_width; yy++)
                                res += input[i * (in_height*in_width*in_channels) + (x + xx) * (in_width*in_channels) + (y + yy) * in_channels + l] * filter_matrix[xx * (filter_width*in_channels*out_channels) + yy * (in_channels*out_channels) + l * out_channels + k];
                        output[i * (out_height*out_width*out_channels) + xx * (out_width*out_channels) + yy * out_channels + k] += res;
                    }
            }
    }
}

//g++ -shared cpp -o cpp.so
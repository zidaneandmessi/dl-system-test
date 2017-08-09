#include <iostream>

using namespace std;

extern "C"
void correlate2d(double *input, double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels)
{
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
                for (int x = 0; x < out_height; x++)
                    for (int y = 0; y < out_width; y++)
                    {
                        double res = 0;
                        for (int xx = 0; xx < filter_height; xx++)
                            for (int yy = 0; yy < filter_width; yy++)
                                res += input[i * (in_height*in_width*in_channels) + (x + xx) * (in_width*in_channels) + (y + yy) * in_channels + l] * filter[xx * (filter_width*in_channels*out_channels) + yy * (in_channels*out_channels) + l * out_channels + k];
                        output[i * (out_height*out_width*out_channels) + x * (out_width*out_channels) + y * out_channels + k] += res;
                    }
}

extern "C"
void correlate2dgrad1(double *input, double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels)
{
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
                for (int x = 0; x < out_height; x++)
                    for (int y = 0; y < out_width; y++)
                    {
                        double res = 0;
                        for (int xx = 0; xx < filter_height; xx++)
                            for (int yy = 0; yy < filter_width; yy++)
                                res += input[i * (in_height*in_width*out_channels) + (x + xx) * (in_width*out_channels) + (y + yy) * out_channels + k] * filter[xx * (filter_width*in_channels*out_channels) + yy * (in_channels*out_channels) + l * out_channels + k];
                        output[i * (out_height*out_width*in_channels) + x * (out_width*in_channels) + y * in_channels + k] += res;
                    }
}

extern "C"
void correlate2dgrad2(double *input, double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels)
{
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
            {
                for (int x = 0; x < out_height; x++)
                    for (int y = 0; y < out_width; y++)
                    {
                        double res = 0;
                        for (int xx = 0; xx < filter_height; xx++)
                            for (int yy = 0; yy < filter_width; yy++)
                                res += input[i * (in_height*in_width*in_channels) + (x + xx) * (in_width*in_channels) + (y + yy) * in_channels + l] * filter[i * (filter_height*filter_width*out_channels) + xx * (filter_width*out_channels) + yy * out_channels + k];
                        output[x * (out_width*in_channels*out_channels) + y * (in_channels*out_channels) + l * out_channels + k] += res;
                    }
            }
}

//g++ -shared cpp -o cpp.so
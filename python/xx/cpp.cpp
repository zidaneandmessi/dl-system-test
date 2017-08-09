#include <iostream>
#include <algorithm>

using namespace std;

extern "C"
void correlate2d(double *input, const double *origin, const double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels, int padding)
{
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < out_height; x++)
            for (int y = 0; y < out_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * (in_height*in_width*in_channels) + (padding + x) * (in_width*in_channels) + (padding + y) * in_channels + k] = origin[i * (out_height*out_width*in_channels) + x * (out_width*in_channels) + y * in_channels + k];
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
void correlate2dgrad1(double *input, const double *origin, const double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels, int padding)
{
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < out_height; x++)
            for (int y = 0; y < out_width; y++)
                for (int k = 0; k < out_channels; k++)
                    input[i * (in_height*in_width*out_channels) + (padding + x) * (in_width*out_channels) + (padding + y) * out_channels + k] = origin[i * (out_height*out_width*out_channels) + x * (out_width*out_channels) + y * out_channels + k];
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
                        output[i * (out_height*out_width*in_channels) + x * (out_width*in_channels) + y * in_channels + l] += res;
                    }
}

extern "C"
void correlate2dgrad2(double *input, const double *origin, const double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels, int padding)
{
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < filter_height; x++)
            for (int y = 0; y < filter_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * (in_height*in_width*in_channels) + (padding + x) * (in_width*in_channels) + (padding + y) * in_channels + k] = origin[i * (filter_height*filter_width*in_channels) + x * (filter_width*in_channels) + y * in_channels + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
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

extern "C"
void maxpool(double *input, const double *origin, double *output, int batch, int in_height, int in_width, int pool_height, int pool_width, int out_height, int out_width, int in_channels, int stride, int padding)
{
    int origin_height = out_height * stride, origin_width = out_width * stride;
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < origin_height; x++)
            for (int y = 0; y < origin_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * (in_height*in_width*in_channels) + (padding + x) * (in_width*in_channels) + (padding + y) * in_channels + k] = origin[i * (origin_height*origin_width*in_channels) + x * (origin_width*in_channels) + y * in_channels + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < in_channels; k++)
            for (int x = 0; x < out_height; x++)
                for (int y = 0; y < out_width; y++)
                {
                    double mx = -1e18;
                    for (int xx = 0; xx < pool_height; xx++)
                        for (int yy = 0; yy < pool_width; yy++)
                            mx = max(mx, input[i * (in_height*in_width*in_channels) + (x * stride + xx) * (in_width*in_channels) + (y * stride + yy) * in_channels + k]);
                    output[i * (out_height*out_width*in_channels) + x * (out_width*in_channels) + y * in_channels + k] = mx;
                }
}

extern "C"
void maxpoolgrad(double *input, const double *gradient, const double *origin, double *ans, double *output, int batch, int in_height, int in_width, int grad_height, int grad_width, int pool_height, int pool_width, int out_height, int out_width, int in_channels, int stride, int padding)
{
    int origin_height = grad_height * stride, origin_width = grad_width * stride;
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < origin_height; x++)
            for (int y = 0; y < origin_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * (in_height*in_width*in_channels) + (padding + x) * (in_width*in_channels) + (padding + y) * in_channels + k] = origin[i * (origin_height*origin_width*in_channels) + x * (origin_width*in_channels) + y * in_channels + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < in_channels; k++)
            for (int x = 0; x < grad_height; x++)
                for (int y = 0; y < grad_width; y++)
                {
                    double mx = -1e18;
                    for (int xx = 0; xx < pool_height; xx++)
                        for (int yy = 0; yy < pool_width; yy++)
                            mx = max(mx, input[i * (in_height*in_width*in_channels) + (x * stride + xx) * (in_width*in_channels) + (y * stride + yy) * in_channels + k]);
                    for (int xx = 0; xx < pool_height; xx++)
                        for (int yy = 0; yy < pool_width; yy++)
                            if (input[i * (in_height*in_width*in_channels) + (x * stride + xx) * (in_width*in_channels) + (y * stride + yy) * in_channels + k] == mx)
                                ans[i * (in_height*in_width*in_channels) + (x * stride + xx) * (in_width*in_channels) + (y * stride + yy) * in_channels + k] = gradient[i * (grad_height*grad_width*in_channels) + x * (grad_width*in_channels) + y * in_channels + k];
                            else
                                ans[i * (in_height*in_width*in_channels) + (x * stride + xx) * (in_width*in_channels) + (y * stride + yy) * in_channels + k] = 0;
                }
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < out_height; x++)
            for (int y = 0; y < out_width; y++)
                for (int k = 0; k < in_channels; k++)
                    output[i * (out_height*out_width*in_channels) + x * (out_width*in_channels) + y * in_channels + k] = ans[i * (in_height*in_width*in_channels) + (padding + x) * (in_width*in_channels) + (padding + y) * in_channels + k];
}

// g++ ./python/xx/cpp.cpp -fPIC -shared -o ./python/xx/cpp.so
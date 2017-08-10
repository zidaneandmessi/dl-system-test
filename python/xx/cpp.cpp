double max(double a, double b)
{
    if (a > b) return a;
    return b;
}

extern "C"
void correlate2d(double *input, const double *origin, const double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels, int padding)
{
    int og[3] = {out_height*out_width*in_channels, out_width*in_channels, in_channels};
    int in[3] = {in_height*in_width*in_channels, in_width*in_channels, in_channels};
    int fi[3] = {filter_width*in_channels*out_channels, in_channels*out_channels, out_channels};
    int ou[3] = {out_height*out_width*out_channels, out_width*out_channels, out_channels};
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < out_height; x++)
            for (int y = 0; y < out_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * in[0] + (padding + x) * in[1] + (padding + y) * in[2] + k] = origin[i * og[0] + x * og[1] + y * og[2] + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
                for (int x = 0; x < out_height; x++)
                    for (int y = 0; y < out_width; y++)
                    {
                        double res = 0;
                        for (int xx = 0; xx < filter_height; xx++)
                            for (int yy = 0; yy < filter_width; yy++)
                                res += input[i * in[0] + (x + xx) * in[1] + (y + yy) * in[2] + l] * filter[xx * fi[0] + yy * fi[1] + l * fi[2] + k];
                        output[i * ou[0] + x * ou[1] + y * ou[2] + k] += res;
                    }
}

extern "C"
void correlate2dgrad1(double *input, const double *origin, const double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels, int padding)
{
    int og[3] = {out_height*out_width*out_channels, out_width*out_channels, out_channels};
    int in[3] = {in_height*in_width*out_channels, in_width*out_channels, out_channels};
    int fi[3] = {filter_width*in_channels*out_channels, in_channels*out_channels, out_channels};
    int ou[3] = {out_height*out_width*in_channels, out_width*in_channels, in_channels};
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < out_height; x++)
            for (int y = 0; y < out_width; y++)
                for (int k = 0; k < out_channels; k++)
                    input[i * in[0] + (padding + x) * in[1] + (padding + y) * in[2] + k] = origin[i * og[0] + x * og[1] + y * og[2] + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
                for (int x = 0; x < out_height; x++)
                    for (int y = 0; y < out_width; y++)
                    {
                        double res = 0;
                        for (int xx = 0; xx < filter_height; xx++)
                            for (int yy = 0; yy < filter_width; yy++)
                                res += input[i * in[0] + (x + xx) * in[1] + (y + yy) * in[2] + k] * filter[xx * fi[0] + yy * fi[1] + l * fi[2] + k];
                        output[i * ou[0] + x * ou[1] + y * ou[2] + l] += res;
                    }
}

extern "C"
void correlate2dgrad2(double *input, const double *origin, const double *filter, double *output, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels, int padding)
{
    int og[3] = {filter_height*filter_width*in_channels, filter_width*in_channels, in_channels};
    int in[3] = {in_height*in_width*in_channels, in_width*in_channels, in_channels};
    int fi[3] = {filter_height*filter_width*out_channels, filter_width*out_channels, out_channels};
    int ou[3] = {out_width*in_channels*out_channels, in_channels*out_channels, out_channels};
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < filter_height; x++)
            for (int y = 0; y < filter_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * in[0] + (padding + x) * in[1] + (padding + y) * in[2] + k] = origin[i * og[0] + x * og[1] + y * og[2] + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < out_channels; k++)
            for (int l = 0; l < in_channels; l++)
                for (int x = 0; x < out_height; x++)
                    for (int y = 0; y < out_width; y++)
                    {
                        double res = 0;
                        for (int xx = 0; xx < filter_height; xx++)
                            for (int yy = 0; yy < filter_width; yy++)
                                res += input[i * in[0] + (x + xx) * in[1] + (y + yy) * in[2] + l] * filter[i * fi[0] + xx * fi[1] + yy * fi[2] + k];
                        output[x * ou[0] + y * ou[1] + l * ou[2] + k] += res;
                    }
}

extern "C"
void maxpool(double *input, const double *origin, double *output, int batch, int in_height, int in_width, int pool_height, int pool_width, int out_height, int out_width, int in_channels, int stride, int padding)
{
    int origin_height = out_height * stride, origin_width = out_width * stride;
    int in[3] = {in_height*in_width*in_channels, in_width*in_channels, in_channels};
    int og[3] = {origin_height*origin_width*in_channels, origin_width*in_channels, in_channels};
    int ou[3] = {out_height*out_width*in_channels, out_width*in_channels, in_channels};
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < origin_height; x++)
            for (int y = 0; y < origin_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * in[0] + (padding + x) * in[1] + (padding + y) * in[2] + k] = origin[i * og[0] + x * og[1] + y * og[2] + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < in_channels; k++)
            for (int x = 0; x < out_height; x++)
                for (int y = 0; y < out_width; y++)
                {
                    double mx = -1e18;
                    for (int xx = 0; xx < pool_height; xx++)
                        for (int yy = 0; yy < pool_width; yy++)
                            mx = max(mx, input[i * in[0] + (x * stride + xx) * in[1] + (y * stride + yy) * in[2] + k]);
                    output[i * ou[0] + x * ou[1] + y * ou[2] + k] = mx;
                }
}

extern "C"
void maxpoolgrad(double *input, const double *gradient, const double *origin, double *ans, double *output, int batch, int in_height, int in_width, int grad_height, int grad_width, int pool_height, int pool_width, int out_height, int out_width, int in_channels, int stride, int padding)
{
    int origin_height = grad_height * stride, origin_width = grad_width * stride;
    int in[3] = {in_height*in_width*in_channels, in_width*in_channels, in_channels};
    int og[3] = {origin_height*origin_width*in_channels, origin_width*in_channels, in_channels};
    int gr[3] = {grad_height*grad_width*in_channels, grad_width*in_channels, in_channels};
    int ou[3] = {out_height*out_width*in_channels, out_width*in_channels, in_channels};
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < origin_height; x++)
            for (int y = 0; y < origin_width; y++)
                for (int k = 0; k < in_channels; k++)
                    input[i * in[0] + (padding + x) * in[1] + (padding + y) * in[2] + k] = origin[i * og[0] + x * og[1] + y * og[2] + k];
    for (int i = 0; i < batch; i++)
        for (int k = 0; k < in_channels; k++)
            for (int x = 0; x < grad_height; x++)
                for (int y = 0; y < grad_width; y++)
                {
                    double mx = -1e18;
                    for (int xx = 0; xx < pool_height; xx++)
                        for (int yy = 0; yy < pool_width; yy++)
                            mx = max(mx, input[i * in[0] + (x * stride + xx) * in[1] + (y * stride + yy) * in[2] + k]);
                    for (int xx = 0; xx < pool_height; xx++)
                        for (int yy = 0; yy < pool_width; yy++)
                            if (input[i * in[0] + (x * stride + xx) * in[1] + (y * stride + yy) * in[2] + k] == mx)
                                ans[i * in[0] + (x * stride + xx) * in[1] + (y * stride + yy) * in[2] + k] = gradient[i * gr[0] + x * gr[1] + y * gr[2] + k];
                            else
                                ans[i * in[0] + (x * stride + xx) * in[1] + (y * stride + yy) * in[2] + k] = 0;
                }
    for (int i = 0; i < batch; i++)
        for (int x = 0; x < out_height; x++)
            for (int y = 0; y < out_width; y++)
                for (int k = 0; k < in_channels; k++)
                    output[i * ou[0] + x * ou[1] + y * ou[2] + k] = ans[i * in[0] + (padding + x) * in[1] + (padding + y) * in[2] + k];
}

// g++ ./python/xx/cpp.cpp -fPIC -shared -o ./python/xx/cpp.so
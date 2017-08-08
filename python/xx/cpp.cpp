#include <cstdlib>

using namespace std;

#define DLL_TEST_API extern "C" __declspec(dllexport)

DLL_TEST_API void correlate2d(float *input, float *filter, float *output_pointer, int batch, int in_height, int in_width, int filter_height, int filter_width, int out_height, int out_width, int in_channels, int out_channels, int padding)
{
    if (padding == 1)
    {
        for (int i = 0; i < batch; i++)
        {
            float *input_matrix = input[i * in_height * in_width * in_channels]
            for (int k = 0; k < out_channels; k++)
                for (int l = 0; l < in_channels; l++)
                {
                    for (int x = 0; x <)
                    filter_matrix = filter_matrix
                }
        }
    }
	// for i in range(batch):
 //        #     input_matrix = input[i, :, :, :]
 //        #     for k in range(out_channels):
 //        #         filter_matrix = filter[:, :, :, k]
 //        #         for l in range(in_channels):
 //        #             output_val[i, :, :, k] += c.correlate2d(input_matrix[:, :, l], filter_matrix[:, :, l], mode = 'same')
}

//g++ -shared cpp -o cpp.so
#include <iostream>
#include <limits>
#include <cstdlib>
#include <chrono>
#include <vector>
#include "conv3x3_8x1x8_dsp.hpp"

static data_t_conv rand_16bit_conv()
{
    return (data_t_conv)(std::rand() & 0x00FF);
}

static acc_t_conv mul_tb_conv(data_t_conv a, data_t_conv b)
{
    return (acc_t_conv)a * (acc_t_conv)b;
}

static data_t_conv read_input_with_padding_tb(
    const std::vector<data_t_conv> &input,
    int input_size,
    int padding,
    int y,
    int x
)
{
    int in_y = y - padding;
    int in_x = x - padding;

    if(in_y < 0 || in_y >= input_size || in_x < 0 || in_x >= input_size)
        return 0;

    return input[in_y * input_size + in_x];
}

static void make_input_conv(std::vector<data_t_conv> &input)
{
    for(size_t i = 0; i < input.size(); i++)
        input[i] = rand_16bit_conv();
}

static void make_weight_conv(std::vector<data_t_conv> &weights)
{
    for(size_t i = 0; i < weights.size(); i++)
        weights[i] = rand_16bit_conv();
}

static void make_golden_conv(
    const std::vector<data_t_conv> &input,
    const std::vector<data_t_conv> &weights,
    std::vector<acc_t_conv> &golden,
    int input_size,
    int output_channels,
    int padding
)
{
    int output_size = input_size + (2 * padding) - KERNEL_SIZE_3X3 + 1;

    for(int oc = 0; oc < output_channels; oc++)
    {
        for(int oy = 0; oy < output_size; oy++)
        {
            for(int ox = 0; ox < output_size; ox++)
            {
                acc_t_conv sum = 0;

                for(int ky = 0; ky < KERNEL_SIZE_3X3; ky++)
                {
                    for(int kx = 0; kx < KERNEL_SIZE_3X3; kx++)
                    {
                        data_t_conv in_val = read_input_with_padding_tb(
                            input,
                            input_size,
                            padding,
                            oy + ky,
                            ox + kx
                        );

                        data_t_conv w_val = weights[oc * KERNEL_ELEMS_3X3 + ky * KERNEL_SIZE_3X3 + kx];
                        sum += mul_tb_conv(in_val, w_val);
                    }
                }

                golden[oc * output_size * output_size + oy * output_size + ox] = sum;
            }
        }
    }
}

int main()
{
    const int input_size = 1024;
    const int output_channels = 8;
    const int padding = 1;
    const int output_size = input_size + (2 * padding) - KERNEL_SIZE_3X3 + 1;

    std::vector<data_t_conv> input(input_size * input_size, 0);
    std::vector<data_t_conv> weights(output_channels * KERNEL_ELEMS_3X3, 0);
    std::vector<acc_t_conv> output_hw(output_channels * output_size * output_size, 0);
    std::vector<acc_t_conv> output_golden(output_channels * output_size * output_size, 0);

    std::srand(1234);

    make_input_conv(input);
    make_weight_conv(weights);

    auto hw_start = std::chrono::high_resolution_clock::now();
    conv3x3_8x1x8_dsp_u64(
        input.data(),
        weights.data(),
        output_hw.data(),
        input_size,
        output_channels,
        padding
    );
    auto hw_end = std::chrono::high_resolution_clock::now();

    auto golden_start = std::chrono::high_resolution_clock::now();
    make_golden_conv(
        input,
        weights,
        output_golden,
        input_size,
        output_channels,
        padding
    );
    auto golden_end = std::chrono::high_resolution_clock::now();

    bool pass = true;
    int mismatch_count = 0;

    for(size_t i = 0; i < output_hw.size(); i++)
    {
        if(output_hw[i] != output_golden[i])
        {
            pass = false;

            if(mismatch_count < 20)
            {
                int oc = (int)(i / (output_size * output_size));
                int rem = (int)(i % (output_size * output_size));
                int oy = rem / output_size;
                int ox = rem % output_size;

                std::cout << "Mismatch oc=" << oc
                          << " oy=" << oy
                          << " ox=" << ox
                          << " hw=" << (unsigned long long)output_hw[i]
                          << " golden=" << (unsigned long long)output_golden[i]
                          << std::endl;
            }

            mismatch_count++;
        }
    }

    auto hw_us = std::chrono::duration_cast<std::chrono::microseconds>(hw_end - hw_start).count();
    auto golden_us = std::chrono::duration_cast<std::chrono::microseconds>(golden_end - golden_start).count();
    unsigned long long total_macs = (unsigned long long)output_channels
                                  * (unsigned long long)output_size
                                  * (unsigned long long)output_size
                                  * (unsigned long long)KERNEL_ELEMS_3X3;

    std::cout << "====================================" << std::endl;
    std::cout << "3x3 Conv Benchmark (8x1 x 8 logical 8x8, DSP version)" << std::endl;
    std::cout << "Input size       : " << input_size << "x" << input_size << std::endl;
    std::cout << "Output size      : " << output_size << "x" << output_size << std::endl;
    std::cout << "Output channels  : " << output_channels << std::endl;
    std::cout << "Padding          : " << padding << std::endl;
    std::cout << "Tile structure   : 8 output channels x 8 output positions" << std::endl;
    std::cout << "PE structure     : 8x1 lane replicated 8 times" << std::endl;
    std::cout << "Kernel flow      : left to right across each 8x1 lane" << std::endl;
    std::cout << "Multiplier       : DSP-directed '*'" << std::endl;
    std::cout << "Total MACs       : " << total_macs << std::endl;
    std::cout << "HW func time     : " << hw_us << " us" << std::endl;
    std::cout << "Golden time      : " << golden_us << " us" << std::endl;
    std::cout << "====================================" << std::endl;

    if(pass)
        std::cout << "PASS" << std::endl;
    else
        std::cout << "FAIL, mismatch count = " << mismatch_count << std::endl;

    std::cout << std::endl;
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    return pass ? 0 : 1;
}

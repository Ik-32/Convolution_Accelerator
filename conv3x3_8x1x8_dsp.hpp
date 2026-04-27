#ifndef CONV3X3_8X1X8_DSP_HPP
#define CONV3X3_8X1X8_DSP_HPP

#include <stdint.h>

constexpr int KERNEL_SIZE_3X3         = 3;
constexpr int KERNEL_ELEMS_3X3        = 9;
constexpr int TILE_OC_8X1X8           = 8;
constexpr int TILE_POS_8X1X8          = 8;
constexpr int MIN_INPUT_SIZE_CONV     = 8;
constexpr int MAX_INPUT_SIZE_CONV     = 1024;
constexpr int MAX_OUTPUT_CHANNELS     = 64;

typedef uint16_t data_t_conv;
typedef uint64_t acc_t_conv;

struct PEConv1D8x1x8
{
    data_t_conv w_reg;
    data_t_conv w_next;
    bool        w_valid_reg;
    bool        w_valid_next;
    acc_t_conv  sum_reg;
    acc_t_conv  sum_next;

    void reset();
    void step(data_t_conv a_in, data_t_conv w_in, bool w_valid_in);
    void commit();
};

extern "C" void conv3x3_8x1x8_dsp_u64(
    const data_t_conv *input,
    const data_t_conv *weights,
    acc_t_conv *output,
    int input_size,
    int output_channels,
    int padding
);

#endif

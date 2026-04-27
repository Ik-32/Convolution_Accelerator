#include "conv3x3_8x1x8_dsp.hpp"

static bool is_power_of_two_conv(int x)
{
#pragma HLS INLINE
    if(x <= 0)
        return false;

    return (x & (x - 1)) == 0;
}

static bool is_valid_conv_config(int input_size, int output_channels, int padding)
{
#pragma HLS INLINE
    if(input_size < MIN_INPUT_SIZE_CONV)
        return false;

    if(input_size > MAX_INPUT_SIZE_CONV)
        return false;

    if(!is_power_of_two_conv(input_size))
        return false;

    if(output_channels <= 0)
        return false;

    if(output_channels > MAX_OUTPUT_CHANNELS)
        return false;

    if((output_channels % TILE_OC_8X1X8) != 0)
        return false;

    if(!(padding == 0 || padding == 1))
        return false;

    int output_size = input_size + (2 * padding) - KERNEL_SIZE_3X3 + 1;
    if(output_size <= 0)
        return false;

    return true;
}

static acc_t_conv dsp_mul_conv(data_t_conv a, data_t_conv b)
{
#pragma HLS INLINE
    acc_t_conv prod = (acc_t_conv)a * (acc_t_conv)b;
#pragma HLS BIND_OP variable=prod op=mul impl=dsp
    return prod;
}

void PEConv1D8x1x8::reset()
{
#pragma HLS INLINE
    w_reg = 0;
    w_next = 0;
    w_valid_reg = false;
    w_valid_next = false;
    sum_reg = 0;
    sum_next = 0;
}

void PEConv1D8x1x8::step(data_t_conv a_in, data_t_conv w_in, bool w_valid_in)
{
#pragma HLS INLINE
    w_next = w_in;
    w_valid_next = w_valid_in;
    sum_next = sum_reg;

    if(w_valid_in)
        sum_next += dsp_mul_conv(a_in, w_in);
}

void PEConv1D8x1x8::commit()
{
#pragma HLS INLINE
    w_reg = w_next;
    w_valid_reg = w_valid_next;
    sum_reg = sum_next;
}

static data_t_conv read_input_with_padding(
    const data_t_conv *input,
    int input_size,
    int padding,
    int y,
    int x
)
{
#pragma HLS INLINE
    int in_y = y - padding;
    int in_x = x - padding;

    if(in_y < 0 || in_y >= input_size || in_x < 0 || in_x >= input_size)
        return 0;

    return input[in_y * input_size + in_x];
}

static void load_weight_tile_8x1x8(
    const data_t_conv *weights,
    data_t_conv weight_tile[TILE_OC_8X1X8][KERNEL_ELEMS_3X3],
    int oc_base
)
{
#pragma HLS INLINE
load_w_oc:
    for(int oc = 0; oc < TILE_OC_8X1X8; oc++)
    {
#pragma HLS UNROLL
    load_w_k:
        for(int k = 0; k < KERNEL_ELEMS_3X3; k++)
        {
#pragma HLS UNROLL
            weight_tile[oc][k] = weights[(oc_base + oc) * KERNEL_ELEMS_3X3 + k];
        }
    }
}

static void clear_pe_tile_8x1x8(PEConv1D8x1x8 pe_tile[TILE_OC_8X1X8][TILE_POS_8X1X8])
{
#pragma HLS INLINE
clear_oc:
    for(int oc = 0; oc < TILE_OC_8X1X8; oc++)
    {
#pragma HLS UNROLL
    clear_pos:
        for(int pos = 0; pos < TILE_POS_8X1X8; pos++)
        {
#pragma HLS UNROLL
            pe_tile[oc][pos].reset();
        }
    }
}

static void compute_conv_tile_8x1x8(
    const data_t_conv *input,
    const data_t_conv weight_tile[TILE_OC_8X1X8][KERNEL_ELEMS_3X3],
    acc_t_conv out_tile[TILE_OC_8X1X8][TILE_POS_8X1X8],
    int input_size,
    int output_size,
    int padding,
    int pos_base
)
{
#pragma HLS INLINE off
    PEConv1D8x1x8 pe_tile[TILE_OC_8X1X8][TILE_POS_8X1X8];
    data_t_conv patch_values[TILE_POS_8X1X8];

#pragma HLS ARRAY_PARTITION variable=pe_tile complete dim=0
#pragma HLS ARRAY_PARTITION variable=patch_values complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=0
#pragma HLS ARRAY_PARTITION variable=out_tile complete dim=0

    clear_pe_tile_8x1x8(pe_tile);

kernel_loop:
    for(int k = 0; k < KERNEL_ELEMS_3X3; k++)
    {
        int ky = k / KERNEL_SIZE_3X3;
        int kx = k % KERNEL_SIZE_3X3;

    load_patch_pos:
        for(int pos = 0; pos < TILE_POS_8X1X8; pos++)
        {
#pragma HLS UNROLL
            int flat_out = pos_base + pos;

            if(flat_out < (output_size * output_size))
            {
                int oy = flat_out / output_size;
                int ox = flat_out % output_size;
                patch_values[pos] = read_input_with_padding(
                    input,
                    input_size,
                    padding,
                    oy + ky,
                    ox + kx
                );
            }
            else
            {
                patch_values[pos] = 0;
            }
        }

    cycle_loop:
        for(int t = 0; t < TILE_POS_8X1X8; t++)
        {
#pragma HLS PIPELINE II=1
            data_t_conv left_weight[TILE_OC_8X1X8];
            bool left_valid[TILE_OC_8X1X8];

#pragma HLS ARRAY_PARTITION variable=left_weight complete dim=0
#pragma HLS ARRAY_PARTITION variable=left_valid complete dim=0

        feed_left_weight:
            for(int oc = 0; oc < TILE_OC_8X1X8; oc++)
            {
#pragma HLS UNROLL
                if(t == 0)
                {
                    left_weight[oc] = weight_tile[oc][k];
                    left_valid[oc] = true;
                }
                else
                {
                    left_weight[oc] = 0;
                    left_valid[oc] = false;
                }
            }

        step_oc:
            for(int oc = 0; oc < TILE_OC_8X1X8; oc++)
            {
#pragma HLS UNROLL
            step_pos:
                for(int pos = 0; pos < TILE_POS_8X1X8; pos++)
                {
#pragma HLS UNROLL
                    data_t_conv w_in;
                    bool w_valid_in;

                    if(pos == 0)
                    {
                        w_in = left_weight[oc];
                        w_valid_in = left_valid[oc];
                    }
                    else
                    {
                        w_in = pe_tile[oc][pos - 1].w_reg;
                        w_valid_in = pe_tile[oc][pos - 1].w_valid_reg;
                    }

                    pe_tile[oc][pos].step(patch_values[pos], w_in, w_valid_in);
                }
            }

        commit_oc:
            for(int oc = 0; oc < TILE_OC_8X1X8; oc++)
            {
#pragma HLS UNROLL
            commit_pos:
                for(int pos = 0; pos < TILE_POS_8X1X8; pos++)
                {
#pragma HLS UNROLL
                    pe_tile[oc][pos].commit();
                }
            }
        }
    }

store_tile:
    for(int oc = 0; oc < TILE_OC_8X1X8; oc++)
    {
#pragma HLS UNROLL
    store_lane:
        for(int pos = 0; pos < TILE_POS_8X1X8; pos++)
        {
#pragma HLS UNROLL
            out_tile[oc][pos] = pe_tile[oc][pos].sum_reg;
        }
    }
}

static void store_output_tile_8x1x8(
    const acc_t_conv out_tile[TILE_OC_8X1X8][TILE_POS_8X1X8],
    acc_t_conv *output,
    int output_size,
    int output_channels,
    int oc_base,
    int pos_base
)
{
#pragma HLS INLINE off
store_oc:
    for(int oc = 0; oc < TILE_OC_8X1X8; oc++)
    {
#pragma HLS UNROLL
    store_pos:
        for(int pos = 0; pos < TILE_POS_8X1X8; pos++)
        {
#pragma HLS UNROLL
            int global_oc = oc_base + oc;
            int flat_out = pos_base + pos;

            if(global_oc < output_channels && flat_out < (output_size * output_size))
                output[global_oc * output_size * output_size + flat_out] = out_tile[oc][pos];
        }
    }
}

extern "C" void conv3x3_8x1x8_dsp_u64(
    const data_t_conv *input,
    const data_t_conv *weights,
    acc_t_conv *output,
    int input_size,
    int output_channels,
    int padding
)
{
#pragma HLS INTERFACE m_axi     port=input   offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi     port=weights offset=slave bundle=gmem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi     port=output  offset=slave bundle=gmem2 max_write_burst_length=64

#pragma HLS INTERFACE s_axilite port=input           bundle=control
#pragma HLS INTERFACE s_axilite port=weights         bundle=control
#pragma HLS INTERFACE s_axilite port=output          bundle=control
#pragma HLS INTERFACE s_axilite port=input_size      bundle=control
#pragma HLS INTERFACE s_axilite port=output_channels bundle=control
#pragma HLS INTERFACE s_axilite port=padding         bundle=control
#pragma HLS INTERFACE s_axilite port=return          bundle=control

    if(!is_valid_conv_config(input_size, output_channels, padding))
        return;

    const int output_size = input_size + (2 * padding) - KERNEL_SIZE_3X3 + 1;
    const int total_positions = output_size * output_size;

    data_t_conv weight_tile[TILE_OC_8X1X8][KERNEL_ELEMS_3X3];
    acc_t_conv  out_tile[TILE_OC_8X1X8][TILE_POS_8X1X8];

#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=0
#pragma HLS ARRAY_PARTITION variable=out_tile complete dim=0

outer_oc_tile:
    for(int oc_base = 0; oc_base < output_channels; oc_base += TILE_OC_8X1X8)
    {
        load_weight_tile_8x1x8(weights, weight_tile, oc_base);

    outer_pos_tile:
        for(int pos_base = 0; pos_base < total_positions; pos_base += TILE_POS_8X1X8)
        {
            compute_conv_tile_8x1x8(
                input,
                weight_tile,
                out_tile,
                input_size,
                output_size,
                padding,
                pos_base
            );

            store_output_tile_8x1x8(
                out_tile,
                output,
                output_size,
                output_channels,
                oc_base,
                pos_base
            );
        }
    }
}

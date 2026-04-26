#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// --- Vitis HW Headers ---
#include "xparameters.h"
#include "xil_cache.h"
#include "xtime_l.h"
#include "xstatus.h"
#include "xconv3x3_sa_8x8_lane_u64.h"

// --- stb_image Library (Image I/O) ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Parameter Settings
#define INPUT_SIZE       1024
#define OUTPUT_CHANNELS  1
#define PADDING          1
#define KERNEL_SIZE_3X3  3
#define KERNEL_ELEMS     9
#define OUTPUT_SIZE      (INPUT_SIZE + (2 * PADDING) - KERNEL_SIZE_3X3 + 1)

#define CONV_DEVICE_ID   XPAR_XCONV3X3_SA_8X8_LANE_U64_0_DEVICE_ID

typedef int16_t data_t;
typedef int64_t acc_t;

// HW Memory Allocation
static data_t input_img[INPUT_SIZE * INPUT_SIZE]                                __attribute__((aligned(64)));
static data_t weights[OUTPUT_CHANNELS * KERNEL_ELEMS]                           __attribute__((aligned(64)));
static acc_t  output_hw[OUTPUT_CHANNELS * OUTPUT_SIZE * OUTPUT_SIZE]            __attribute__((aligned(64)));

static XConv3x3_sa_8x8_lane_u64 do_conv;

// --- Load Image and Initialize ---
static int load_image_and_init(const char* filepath) {
    int width, height, channels;

    printf("[INFO] Loading image... : %s\r\n", filepath);
    unsigned char *img_data = stbi_load(filepath, &width, &height, &channels, 1);

    if (img_data == NULL) {
        printf("[ERROR] Image Not Found or Cannot Open: %s\r\n", filepath);
        return -1;
    }
    printf("[INFO] Image Load Success: %d x %d\r\n", width, height);

    // Copy image data to HW input array
    for (int y = 0; y < INPUT_SIZE; y++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            if (y < height && x < width) {
                input_img[y * INPUT_SIZE + x] = (data_t)img_data[y * width + x];
            } else {
                input_img[y * INPUT_SIZE + x] = 0;
            }
        }
    }
    stbi_image_free(img_data);

    // Set Edge Detection Kernel (Sobel X)
    data_t sobel_x[KERNEL_ELEMS] = {-1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1};
    for(int i = 0; i < KERNEL_ELEMS; i++) {
        weights[i] = sobel_x[i];
    }

    return 0;
}

// --- Save Result to Image File ---
static void save_output_image(const char* filepath) {
    unsigned char* out_img_data = (unsigned char*)malloc(OUTPUT_SIZE * OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE; i++) {
        acc_t val = output_hw[i];
        if (val < 0) val = -val;
        if (val > 255) val = 255;
        out_img_data[i] = (unsigned char)val;
    }

    stbi_write_png(filepath, OUTPUT_SIZE, OUTPUT_SIZE, 1, out_img_data, OUTPUT_SIZE);
    printf("[INFO] Edge Image Saved: %s\r\n", filepath);

    free(out_img_data);
}

// --- HW Convolution Execution ---
static void run_hw_convolution() {
    Xil_DCacheFlushRange((UINTPTR)input_img, sizeof(input_img));
    Xil_DCacheFlushRange((UINTPTR)weights, sizeof(weights));
    Xil_DCacheFlushRange((UINTPTR)output_hw, sizeof(output_hw));

    XConv3x3_sa_8x8_lane_u64_Set_input_r(&do_conv, (u64)(UINTPTR)input_img);
    XConv3x3_sa_8x8_lane_u64_Set_weights(&do_conv, (u64)(UINTPTR)weights);
    XConv3x3_sa_8x8_lane_u64_Set_output_r(&do_conv, (u64)(UINTPTR)output_hw);
    XConv3x3_sa_8x8_lane_u64_Set_input_size(&do_conv, INPUT_SIZE);
    XConv3x3_sa_8x8_lane_u64_Set_output_channels(&do_conv, OUTPUT_CHANNELS);
    XConv3x3_sa_8x8_lane_u64_Set_padding(&do_conv, PADDING);

    XConv3x3_sa_8x8_lane_u64_Start(&do_conv);
    while(!XConv3x3_sa_8x8_lane_u64_IsDone(&do_conv));

    Xil_DCacheInvalidateRange((UINTPTR)output_hw, sizeof(output_hw));
}

int main(void) {
    int status;
    printf("\r\n========================================\r\n");
    printf("FPGA Image Edge Detection\r\n");
    printf("========================================\r\n");

    // Initialize IP
    status = XConv3x3_sa_8x8_lane_u64_Initialize(&do_conv, CONV_DEVICE_ID);
    if(status != XST_SUCCESS) {
        printf("[ERROR] HW IP Init Failed\r\n");
        return -1;
    }

    // *** ENTER YOUR IMAGE PATH HERE ***
    const char* input_image_path = "C:\\Users\\User\\Desktop\\gul.png";

    // Load the image from the specified path
    if (load_image_and_init(input_image_path) != 0) {
        return -1;
    }

    printf("[INFO] HW Conv Start...\r\n");
    run_hw_convolution();
    printf("[INFO] HW Conv Done.\r\n");

    save_output_image("output_edge.png");

    printf("\r\n[SUCCESS] Pipeline Completed.\r\n");
    return 0;
}

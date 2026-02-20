#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_math.h>

typedef ap_axiu<32, 1, 1, 1> axis_t;
#define MAX_WIDTH 1920
#define K_SIZE 3

// Bit-safe float-to-int conversion
union data_conv {
    unsigned int i;
    float f;
};

// =========================================================================
// MODULE 1: CONVOLUTION + ReLU
// =========================================================================
void conv_layer(
    hls::stream<axis_t>& in_stream,
    hls::stream<float>& internal_stream,
    const float* weights_port,
    int width,
    int height
) {
    // Line buffers for sliding window
    static float line_buf[K_SIZE - 1][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buf dim=1 complete
    #pragma HLS RESOURCE variable=line_buf core=RAM_S2P_BRAM

    float window[K_SIZE][K_SIZE];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // Local weight cache
    float kernel[K_SIZE][K_SIZE];
    #pragma HLS ARRAY_PARTITION variable=kernel complete dim=0
    float bias = 0.0f;

    // Load weights via AXI MM
    volatile float* v_ptr = (volatile float*)weights_port;
    bias = v_ptr[0];
    for(int i=0; i<K_SIZE; i++) {
        for(int j=0; j<K_SIZE; j++) {
            #pragma HLS PIPELINE
            kernel[i][j] = v_ptr[1 + (i * K_SIZE + j)];
        }
    }

    // Process image
    pixel_loop_y: for (int y = 0; y < height; y++) {
        pixel_loop_x: for (int x = 0; x < width; x++) {
            #pragma HLS PIPELINE II=1

            // Read pixel
            axis_t packet = in_stream.read();
            data_conv conv_in;
            conv_in.i = packet.data;
            float current_pixel = conv_in.f;

            // Shift sliding window and line buffers
            for (int i = 0; i < K_SIZE; i++)
                for (int j = 0; j < K_SIZE - 1; j++)
                    window[i][j] = window[i][j + 1];

            window[0][K_SIZE-1] = line_buf[0][x];
            window[1][K_SIZE-1] = line_buf[1][x];
            window[2][K_SIZE-1] = current_pixel;

            line_buf[0][x] = line_buf[1][x];
            line_buf[1][x] = current_pixel;

            // Compute Convolution
            float result = 0.0f;
            if (y >= K_SIZE - 1 && x >= K_SIZE - 1) {
                float sum = 0.0f;
                for(int i=0; i<K_SIZE; i++)
                    for(int j=0; j<K_SIZE; j++)
                        sum += window[i][j] * kernel[i][j];

                result = sum + bias;

                // ReLU
                if (result < 0.0f) result = 0.0f;
            }

            // Write to stream (including invalid padding cycles)
            internal_stream.write(result);
        }
    }
}

// =========================================================================
// MODULE 2: MAX POOLING (2x2)
// =========================================================================
void max_pool(
    hls::stream<float>& in_stream,
    hls::stream<axis_t>& out_stream,
    int width,
    int height
) {
    // Buffer for previous row
    static float pool_buf[MAX_WIDTH];
    #pragma HLS RESOURCE variable=pool_buf core=RAM_S2P_BRAM

    // 2x2 pooling window columns
    float window_col_0[2]; 
    float window_col_1[2]; 

    // Read 1-to-1, output 1-to-4
    pool_y: for (int y = 0; y < height; y++) {
        pool_x: for (int x = 0; x < width; x++) {
            #pragma HLS PIPELINE II=1

            float val_in = in_stream.read();
            float val_prev_row = 0.0f;

            if (y % 2 == 0) {
                // Even row: store pixel
                pool_buf[x] = val_in;
            } else {
                // Odd row: retrieve pixel and update window
                val_prev_row = pool_buf[x];

                window_col_0[0] = window_col_1[0]; 
                window_col_0[1] = window_col_1[1]; 

                window_col_1[0] = val_prev_row;    
                window_col_1[1] = val_in;          

                // Output at bottom-right of 2x2 block
                if (x % 2 == 1) {
                    float max_val = window_col_0[0];
                    if (window_col_0[1] > max_val) max_val = window_col_0[1];
                    if (window_col_1[0] > max_val) max_val = window_col_1[0];
                    if (window_col_1[1] > max_val) max_val = window_col_1[1];

                    // Pack AXI-Stream packet
                    axis_t packet;
                    data_conv conv_out;
                    conv_out.f = max_val;
                    packet.data = conv_out.i;

                    // Assert TLAST on the final output pixel
                    if ((x == width - 1) && (y == height - 1)) {
                        packet.last = 1;
                    } else {
                        packet.last = 0;
                    }

                    packet.keep = -1;
                    packet.strb = -1;
                    out_stream.write(packet);
                }
            }
        }
    }
}



// =========================================================================
// TOP LEVEL WRAPPER
// =========================================================================
void dpu_core(
    hls::stream<axis_t>& in_stream,
    hls::stream<axis_t>& out_stream,
    const float* weights_port,
    int width,
    int height
) {
    // AXI Interfaces
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE m_axi port=weights_port offset=slave bundle=gmem depth=1024
    #pragma HLS INTERFACE s_axilite port=width bundle=control
    #pragma HLS INTERFACE s_axilite port=height bundle=control
    #pragma HLS INTERFACE s_axilite port=weights_port bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Enable task-level pipelining (Parallel execution)
    #pragma HLS DATAFLOW

    // FIFO to connect Conv and Pool modules
    hls::stream<float> connect_stream;
    #pragma HLS STREAM variable=connect_stream depth=128

    // Execute modules
    conv_layer(in_stream, connect_stream, weights_port, width, height);
    max_pool(connect_stream, out_stream, width, height);
}

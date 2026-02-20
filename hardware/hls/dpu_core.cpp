#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include "weights.h"

typedef ap_axiu<32, 0, 0, 0> axis_t;

union float_conv {
    float f;
    unsigned int i;
};

template<int CH_IN, int CH_OUT, int H_IN, int W_IN>
void conv_relu_pool_layer(
    const float in_fmap[H_IN][W_IN][CH_IN],
    float out_fmap[H_IN/2][W_IN/2][CH_OUT],
    const float weights[],
    const float biases[]
) {
    for (int co = 0; co < CH_OUT; co++) {
        for (int h = 0; h < H_IN / 2; h++) {
            for (int w = 0; w < W_IN / 2; w++) {

                float max_val = -1e38f;

                for (int ph = 0; ph < 2; ph++) {
                    for (int pw = 0; pw < 2; pw++) {
                        int in_h = h * 2 + ph;
                        int in_w = w * 2 + pw;
                        float conv_sum = biases[co];

                        for (int ci = 0; ci < CH_IN; ci++) {
                            for (int kh = 0; kh < 3; kh++) {
                                for (int kw = 0; kw < 3; kw++) {
                                    #pragma HLS PIPELINE II=1

                                    int r_h = in_h + kh - 1;
                                    int r_w = in_w + kw - 1;
                                    float pixel = 0.0f;

                                    if (r_h >= 0 && r_h < H_IN && r_w >= 0 && r_w < W_IN) {
                                        pixel = in_fmap[r_h][r_w][ci];
                                    }

                                    int w_idx = co * (CH_IN * 9) + ci * 9 + kh * 3 + kw;
                                    conv_sum += pixel * weights[w_idx];
                                }
                            }
                        }

                        float relu_val = (conv_sum > 0.0f) ? conv_sum : 0.0f;
                        if (relu_val > max_val) max_val = relu_val;
                    }
                }
                out_fmap[h][w][co] = max_val;
            }
        }
    }
}

void classifier_head(const float in_fmap[4][4][64], float logits[10]) {
    for (int out_c = 0; out_c < 10; out_c++) {
        float sum = fc_b[out_c];
        int flat_idx = 0;

        for (int c = 0; c < 64; c++) {
            for (int h = 0; h < 4; h++) {
                for (int w = 0; w < 4; w++) {
                    #pragma HLS PIPELINE II=1
                    sum += in_fmap[h][w][c] * fc_w[out_c * 1024 + flat_idx];
                    flat_idx++;
                }
            }
        }
        logits[out_c] = sum;
    }
}

void cnn_accelerator(hls::stream<axis_t>& img_in, hls::stream<axis_t>& logits_out) {
    #pragma HLS INTERFACE axis port=img_in
    #pragma HLS INTERFACE axis port=logits_out
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS BIND_STORAGE variable=layer1_w type=rom_1p impl=bram
    #pragma HLS BIND_STORAGE variable=layer1_b type=rom_1p impl=bram
    #pragma HLS BIND_STORAGE variable=layer2_w type=rom_1p impl=bram
    #pragma HLS BIND_STORAGE variable=layer2_b type=rom_1p impl=bram
    #pragma HLS BIND_STORAGE variable=layer3_w type=rom_1p impl=bram
    #pragma HLS BIND_STORAGE variable=layer3_b type=rom_1p impl=bram
    #pragma HLS BIND_STORAGE variable=fc_w     type=rom_1p impl=bram
    #pragma HLS BIND_STORAGE variable=fc_b     type=rom_1p impl=bram

    static float fmap0[32][32][3];
    static float fmap1[16][16][16];
    static float fmap2[8][8][32];
    static float fmap3[4][4][64];
    float logits[10];

    for (int h = 0; h < 32; h++) {
        for (int w = 0; w < 32; w++) {
            for (int c = 0; c < 3; c++) {
                #pragma HLS PIPELINE II=1
                axis_t pkt = img_in.read();
                float_conv conv; 
                conv.i = (unsigned int)pkt.data;
                fmap0[h][w][c] = conv.f;
            }
        }
    }

    conv_relu_pool_layer<3, 16, 32, 32>(fmap0, fmap1, layer1_w, layer1_b);
    conv_relu_pool_layer<16, 32, 16, 16>(fmap1, fmap2, layer2_w, layer2_b);
    conv_relu_pool_layer<32, 64, 8, 8>(fmap2, fmap3, layer3_w, layer3_b);
    classifier_head(fmap3, logits);

    for (int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        axis_t pkt_out;
        float_conv conv; 
        conv.f = logits[i];
        pkt_out.data = conv.i;
        pkt_out.keep = -1;
        pkt_out.strb = -1;
        pkt_out.last = (i == 9) ? 1 : 0;
        logits_out.write(pkt_out);
    }
}

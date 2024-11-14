#ifndef __DEEPGUIDEDFILTER__H__
#define __DEEPGUIDEDFILTER__H__

#include "ggml_engine.h"
#include "ggml_nn.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>

ggml_tensor_t *ggml_box_filter(struct ggml_context* ctx, ggml_tensor_t *x, int r);

#pragma GCC diagnostic ignored "-Wformat-truncation"

// ggml_set_name(out, "xxxx_test");
// ggml_set_output(out);

/*
 BoxFilter() */

// CPU OK, CUDA NOK ----------------------------
struct BoxFilter {
    int r = 1;

    void create_weight_tensors(ggml_context_t* ctx) {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char *prefix) {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = ggml_nn_arange(ctx, x);
        ggml_tensor_t* y = ggml_box_filter(ctx, x, r);

        // ggml_set_name(y, "xxxx_test");
        // ggml_set_output(y);

        // Info: ********************** xxxx_test Tensor: 1x3x85x128
        // min: 0.0079, max: 8.9523, mean: 2.9688

        // Info: ********************** xxxx_test Tensor: 1x3x85x128
        // min: 0.0079, max: 8.9642, mean: 4.4413

        // tensor [xxxx_test] size: [1, 3, 85, 128], min: 0.007904, max: 8.964172, mean: 4.441317
        // tensor [xxxx_test] size: [1, 3, 85, 128], min: 0.007904, max: 8.964172, mean: 4.441317
        // tensor [xxxx_test] size: [1, 3, 85, 128], min: 0.007904, max: 8.964172, mean: 4.441317
        // tensor [xxxx_test] size: [1, 3, 85, 128], min: 0.007904, max: 8.964172, mean: 4.441317
        // tensor [xxxx_test] size: [1, 3, 85, 128], min: 0.007904, max: 8.964172, mean: 4.441317

        return y;
    }
};

/*
 FastGuidedFilter(
  (boxfilter): BoxFilter()
) */

// CPU OK, CUDA NOK --------------------------
struct FastGuidedFilter {
    int r = 1;
    float eps = 1e-08;

    struct BoxFilter boxfilter;

    void create_weight_tensors(ggml_context_t* ctx) {
        GGML_UNUSED(ctx);
    }

    void setup_weight_names(const char *prefix) {
        GGML_UNUSED(prefix);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x1, ggml_tensor_t* x2, ggml_tensor_t* x3) {
        // x1 = ggml_nn_arange(ctx, x1);
        // x2 = ggml_nn_arange(ctx, x2);
        // x3 = ggml_nn_arange(ctx, x3);


        // n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()
        int W3 = (int)x3->ne[0];
        int H3 = (int)x3->ne[1];
        int C3 = (int)x3->ne[2];
        int B3 = (int)x3->ne[3];
        ggml_tensor_dump("--------x3", x3);

        ggml_tensor_t *N = ggml_dup(ctx, x1);
        N = ggml_clamp(ctx, N, 1.0, 1.0); // ggml_constant(ctx, N, 1.0);
        ggml_tensor_dump("N1", N);
        N = boxfilter.forward(ctx, N);
        ggml_tensor_dump("N2", N);

        ggml_tensor_t *mean_x = boxfilter.forward(ctx, x1);
        ggml_tensor_dump("mean_x1", mean_x);
        mean_x = ggml_div(ctx, mean_x, N);
        ggml_tensor_dump("mean_x2", mean_x);

        ggml_tensor_t *mean_y = boxfilter.forward(ctx, x2);
        ggml_tensor_dump("mean_y1", mean_y);
        mean_y = ggml_div(ctx, mean_y, N);
        ggml_tensor_dump("mean_y2", mean_y);

        ggml_tensor_t *mean_xy = ggml_mul(ctx, mean_x, mean_y);
        ggml_tensor_dump("mean_xy", mean_y);

        // ## cov_xy
        // cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        ggml_tensor_t *cov_xy = ggml_mul(ctx, x1, x2);
        ggml_tensor_dump("cov_xy1", cov_xy);
        cov_xy = boxfilter.forward(ctx, cov_xy);
        ggml_tensor_dump("cov_xy2", cov_xy);
        cov_xy = ggml_div(ctx, cov_xy, N);
        ggml_tensor_dump("cov_xy3", cov_xy);
        cov_xy = ggml_sub(ctx, cov_xy, mean_xy); 
        ggml_tensor_dump("cov_xy4", cov_xy);

        // ## var_x
        // var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x
        ggml_tensor_t *var_x = ggml_mul(ctx, x1, x1);
        ggml_tensor_dump("var_x1", var_x);
        var_x = boxfilter.forward(ctx, var_x);
        ggml_tensor_dump("var_x2", var_x);
        var_x = ggml_div(ctx, var_x, N);
        ggml_tensor_dump("var_x3", var_x);
        var_x = ggml_sub(ctx, var_x, mean_xy);
        ggml_tensor_dump("var_x4", var_x);
        var_x = ggml_add_constant(ctx, var_x, eps);
        ggml_tensor_dump("var_x5", var_x);

        // ## A
        // A = cov_xy / (var_x + self.eps)

        // ## b
        // b = mean_y - A * mean_x
        ggml_tensor_t *A = ggml_div(ctx, cov_xy, var_x);
        ggml_tensor_dump("A", A);
        ggml_tensor_t *t = ggml_mul(ctx, A, mean_x);
        ggml_tensor_dump("t", t);
        ggml_tensor_t *b = ggml_sub(ctx, mean_y, t);
        ggml_tensor_dump("b", b);

        ggml_tensor_t *mean_A = ggml_upscale_ext(ctx, A, W3, H3, C3, B3);
        ggml_tensor_dump("mean_A", mean_A);

        ggml_tensor_t *mean_b = ggml_upscale_ext(ctx, b, W3, H3, C3, B3);
        ggml_tensor_dump("mean_b", mean_b);

        // ## mean_A; mean_b
        // mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=False)

        // mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=False)

        // output = mean_A * hr_x + mean_b
        ggml_tensor_dump("x3", x3);
        ggml_tensor_t *output = ggml_mul(ctx, mean_A, x3);
        ggml_tensor_dump("output1", output);
        output = ggml_add(ctx, output, mean_b);
        ggml_tensor_dump("output2", output);
        output = ggml_clamp(ctx, output, 0.0, 1.0);

        output = ggml_cont(ctx, output);

        ggml_set_name(output, "xxxx_test");
        ggml_set_output(output);

        // Info: ********************** xxxx_test Tensor: 1x3x341x512
        // min: 0.0000, max: 1.0000, mean: 0.5000
        // Info: ********************** xxxx_test Tensor: 1x3x341x512
        // min: 0.0000, max: nan, mean: nan

        // tensor [xxxx_test] size: [1, 3, 341, 512], min: 0.001427, max: 0.998522, mean: 0.499993

        return output;
        // return output.clamp(0.0, 1.0)
    }
};

/*
 AdaptiveNorm(
  (bn): BatchNorm2d(15, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
) */

struct AdaptiveNorm {
    int num_features = 24;

    ggml_tensor_t* w_0;
    ggml_tensor_t* w_1;
    struct BatchNorm2d bn;

    void create_weight_tensors(ggml_context_t* ctx) {
        w_0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        w_1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        // tensor [self.w_0] size: [1], 1.714943
        // tensor [self.w_1] size: [1], 1.024461

        bn.num_features = num_features;
        bn.eps = 0.001;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "w_0");
        ggml_set_name(w_0, s);

        snprintf(s, sizeof(s), "%s%s", prefix, "w_1");
        ggml_set_name(w_1, s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // y = self.w_0 * x + self.w_1 * self.bn(x)
        // ggml_tensor_dump("xx0", x);

        // x = ggml_nn_arange(ctx, x);
        ggml_tensor_t *x1 = ggml_mul(ctx, x, w_0);
        ggml_tensor_t *x2 = ggml_mul(ctx, bn.forward(ctx, x), w_1);
        x = ggml_add(ctx, x1, x2);

        // x = bn.forward(ctx, x);
        // x = ggml_mul(ctx, x, w_1);
        // x = ggml_add(ctx, x, w_0);
        x = ggml_cont(ctx, x);

        // ggml_set_name(x, "xxxx_test");
        // ggml_set_output(x);

        // Info: ********************** xxxx_test Tensor: 1x15x341x512
        // min: -0.4744, max: 19.2170, mean: 5.3924

        // tensor [xxxx_test] size: [1, 15, 341, 512], min: -2.373173, max: 19.225903, mean: 5.180735


    	return x;
    }
};

// ------------------------------------------------------------------
struct GuidedMap {
    struct Conv2d conv0;
    struct AdaptiveNorm bn1;
    struct Conv2d conv3;

    // (guided_map): Sequential(
    //   (0): Conv2d(3, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
    //   (1): AdaptiveNorm(
    //     (bn): BatchNorm2d(15, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
    //   )
    //   (2): LeakyReLU(negative_slope=0.2, inplace=True)
    //   (3): Conv2d(15, 3, kernel_size=(1, 1), stride=(1, 1))
    // )
    void create_weight_tensors(ggml_context_t* ctx) {
        conv0.in_channels = 3;
        conv0.out_channels = 15;
        conv0.kernel_size = {1, 1};
        conv0.stride = { 1, 1 };
        conv0.has_bias = false;
        conv0.create_weight_tensors(ctx);

        bn1.num_features = 15;
        bn1.create_weight_tensors(ctx);

        conv3.in_channels = 15;
        conv3.out_channels = 3;
        conv3.kernel_size = {1, 1};
        conv3.stride = { 1, 1 };
        conv3.has_bias = true;
        conv3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        conv0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        bn1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "3.");
        conv3.setup_weight_names(s);

    }

// (guided_map): Sequential(
//   (0): Conv2d(3, 15, kernel_size=(1, 1), stride=(1, 1), bias=False)
//   (1): AdaptiveNorm(
//     (bn): BatchNorm2d(15, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
//   )
//   (2): LeakyReLU(negative_slope=0.2, inplace=True)
//   (3): Conv2d(15, 3, kernel_size=(1, 1), stride=(1, 1))
// )

    // GGML_API struct ggml_tensor * ggml_leaky_relu(
    //         struct ggml_context * ctx,
    //         struct ggml_tensor  * a, float negative_slope, bool inplace);

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = ggml_nn_arange(ctx, x);

        // ggml_tensor_dump("x1", x);
        // xxxx_debug
        x = conv0.forward(ctx, x);
        // ggml_tensor_dump("x2", x);
        x = bn1.forward(ctx, x);
        // ggml_tensor_dump("x3", x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        // ggml_tensor_dump("x4", x);
        x = conv3.forward(ctx, x);
        // ggml_tensor_dump("x5", x);

        // Info: ********************** xxxx_test Tensor: 1x3x341x512
        // min: -23.5009, max: 25.5696, mean: -0.5659

        return x;
    }
};

// ---------------------------------------------------------
struct LowGuidedMap {
    struct Conv2d l_0;
    struct AdaptiveNorm l_1;
    struct Conv2d l_3;
    struct AdaptiveNorm l_4;
    struct Conv2d l_6;
    struct AdaptiveNorm l_7;
    struct Conv2d l_9;
    struct AdaptiveNorm l_10;
    struct Conv2d l_12;
    struct AdaptiveNorm l_13;
    struct Conv2d l_15;
    struct AdaptiveNorm l_16;
    struct Conv2d l_18;

    void create_weight_tensors(ggml_context_t* ctx) {
        // (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        l_0.in_channels = 3;
        l_0.out_channels = 24;
        l_0.kernel_size = {3, 3};
        l_0.stride = { 1, 1 };
        l_0.padding = { 1, 1 };
        l_0.has_bias = false;
        l_0.create_weight_tensors(ctx);

        // (3): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        l_3.in_channels = 24;
        l_3.out_channels = 24;
        l_3.kernel_size = {3, 3};
        l_3.stride = { 1, 1 };
        l_3.padding = { 2, 2 };
        l_3.dilation = { 2, 2 };
        l_3.has_bias = false;
        l_3.create_weight_tensors(ctx);

        // (6): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        l_6.in_channels = 24;
        l_6.out_channels = 24;
        l_6.kernel_size = {3, 3};
        l_6.stride = { 1, 1 };
        l_6.padding = { 4, 4 };
        l_6.dilation = { 4, 4 };
        l_6.has_bias = false;
        l_6.create_weight_tensors(ctx);

        // (9): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), bias=False)
        l_9.in_channels = 24;
        l_9.out_channels = 24;
        l_9.kernel_size = {3, 3};
        l_9.stride = { 1, 1 };
        l_9.padding = { 8, 8 };
        l_9.dilation = { 8, 8 };
        l_9.has_bias = false;
        l_9.create_weight_tensors(ctx);

        // (12): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(16, 16), dilation=(16, 16), bias=False)
        l_12.in_channels = 24;
        l_12.out_channels = 24;
        l_12.kernel_size = {3, 3};
        l_12.stride = { 1, 1 };
        l_12.padding = { 16, 16 };
        l_12.dilation = { 16, 16 };
        l_12.has_bias = false;
        l_12.create_weight_tensors(ctx);

        // (15): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        l_15.in_channels = 24;
        l_15.out_channels = 24;
        l_15.kernel_size = {3, 3};
        l_15.stride = { 1, 1 };
        l_15.padding = { 1, 1 };
        l_15.dilation = { 1, 1 };
        l_15.has_bias = false;
        l_15.create_weight_tensors(ctx);

        // (18): Conv2d(24, 3, kernel_size=(1, 1), stride=(1, 1))
        l_18.in_channels = 24;
        l_18.out_channels = 3;
        l_18.kernel_size = {1, 1};
        l_18.stride = { 1, 1 };
        l_18.padding = { 0, 0 };
        l_18.dilation = { 1, 1 };
        l_18.has_bias = true;
        l_18.create_weight_tensors(ctx);

        // (1): AdaptiveNorm(
        //   (bn): BatchNorm2d(24, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
        // )
        l_1.num_features = 24;
        l_1.create_weight_tensors(ctx);

        // (4): AdaptiveNorm(
        //   (bn): BatchNorm2d(24, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
        // )
        l_4.num_features = 24;
        l_4.create_weight_tensors(ctx);

        // (7): AdaptiveNorm(
        //   (bn): BatchNorm2d(24, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
        // )
        l_7.num_features = 24;
        l_7.create_weight_tensors(ctx);

        // (10): AdaptiveNorm(
        //   (bn): BatchNorm2d(24, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
        // )
        l_10.num_features = 24;
        l_10.create_weight_tensors(ctx);

        // (13): AdaptiveNorm(
        //   (bn): BatchNorm2d(24, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
        // )
        l_13.num_features = 24;
        l_13.create_weight_tensors(ctx);

        // (16): AdaptiveNorm(
        //   (bn): BatchNorm2d(24, eps=0.001, momentum=0.999, affine=True, track_running_stats=True)
        // )
        l_16.num_features = 24;
        l_16.create_weight_tensors(ctx);

    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "0.");
        l_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "1.");
        l_1.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "2.");
        // l_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "3.");
        l_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "4.");
        l_4.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "5.");
        // l_5.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "6.");
        l_6.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "7.");
        l_7.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "8.");
        // l_8.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "9.");
        l_9.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "10.");
        l_10.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "11.");
        // l_11.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "12.");
        l_12.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "13.");
        l_13.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "14.");
        // l_14.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "15.");
        l_15.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "16.");
        l_16.setup_weight_names(s);
        // snprintf(s, sizeof(s), "%s%s", prefix, "17.");
        // l_17.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "18.");
        l_18.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = ggml_nn_arange(ctx, x);

        x = l_0.forward(ctx, x);
        x = l_1.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        x = l_3.forward(ctx, x);
        x = l_4.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        x = l_6.forward(ctx, x);
        x = l_7.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        x = l_9.forward(ctx, x);
        x = l_10.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        x = l_12.forward(ctx, x);
        x = l_13.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        x = l_15.forward(ctx, x);
        x = l_16.forward(ctx, x);
        x = ggml_leaky_relu(ctx, x, 0.2, true /*inplace*/);
        x = l_18.forward(ctx, x);

        // x = ggml_cont(ctx, x);
        // ggml_set_name(x, "xxxx_test");
        // ggml_set_output(x);

        // Info: ********************** xxxx_test Tensor: 1x3x85x128
        // min: -0.3747, max: 1.1702, mean: 0.4043
        // tensor [xxxx_test] size: [1, 3, 85, 128], min: -0.374559, max: 1.170218, mean: 0.403946

        return x;
    }
};


struct DeepGuidedFilter : GGMLNetwork {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 1024;
    int MAX_TIMES = 1;

    // network params
    // self.lr = build_lr_net()
    // self.gf = FastGuidedFilter(radius, eps)

    // self.guided_map = nn.Sequential(
    //     nn.Conv2d(3, 15, 1, bias=False), 
    //     AdaptiveNorm(15), 
    //     nn.LeakyReLU(0.2, inplace=True), 
    //     nn.Conv2d(15, 3, 1)
    // )

    struct LowGuidedMap lr;
    struct FastGuidedFilter gf;
    struct GuidedMap guided_map;

    void create_weight_tensors(ggml_context_t* ctx) {
        lr.create_weight_tensors(ctx);
        gf.create_weight_tensors(ctx);
        guided_map.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "lr.");
        lr.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "gf.");
        gf.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "guided_map.");
        guided_map.setup_weight_names(s);
    }

    // B, C, H, W = x_hr.size()

    // x_lr = F.interpolate(x_hr, (H//4, W//4), mode="bilinear", align_corners=False)

    // return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))


    ggml_tensor_t* forward(struct ggml_context* ctx, int eng_argc, ggml_tensor_t* eng_argv[])
    {
        ggml_tensor_t* x = eng_argv[0];
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        ggml_tensor_dump("start_x", x);

        ggml_tensor_t *x_lr = ggml_upscale_ext(ctx, x, (W/4), (H/4), C, B);
        // ggml_tensor_dump("x_lr", x_lr);
        ggml_set_name(x_lr, "x_0");
        ggml_set_output(x_lr);

        ggml_tensor_t *x_1 = guided_map.forward(ctx, x_lr);
        // ggml_tensor_dump("x_1", x_1);
        x_1 = ggml_cont(ctx, x_1);
        ggml_set_name(x_1, "x_1");
        ggml_set_output(x_1);

        ggml_tensor_t *x_2 = lr.forward(ctx, x_lr);
        // ggml_tensor_dump("x_2", x_2);
        x_2 = ggml_cont(ctx, x_2);
        ggml_set_name(x_2, "x_2");
        ggml_set_output(x_2);

        ggml_tensor_dump("---> x", x);
        ggml_tensor_t *x_3 = guided_map.forward(ctx, x);
        // ggml_tensor_dump("x_3", x_3);
        x_3 = ggml_cont(ctx, x_3);
        ggml_set_name(x_3, "x_3");
        ggml_set_output(x_3);

        ggml_tensor_t *y = gf.forward(ctx, x_1, x_2, x_3);
        // ggml_tensor_dump("y", y);
        y = ggml_cont(ctx, y);
        ggml_set_name(y, "x_4");
        ggml_set_output(y);

        // Info: ********************** x_0 Tensor: 1x3x85x128
        // min: 0.0549, max: 0.7059, mean: 0.3521

        // Info: ********************** x_1 Tensor: 1x3x85x128
        // min: -14.8681, max: 19.6397, mean: 1.3930

        // Info: ********************** x_2 Tensor: 1x3x85x128
        // min: -inf, max: -inf, mean: nan
        // Info: ********************** x_3 Tensor: 1x3x341x512
        // min: -19.6884, max: 20.3428, mean: 1.4013
        // Info: ********************** x_4 Tensor: 1x3x341x512
        // min: nan, max: nan, mean: nan


        // tensor [x_0] size: [1, 3, 85, 128], min: 0.072457, max: 0.713841, mean: 0.350823
        // tensor [x_1] size: [1, 3, 85, 128], min: -29.092403, max: 25.128359, mean: 0.740151
        // tensor [x_2] size: [1, 3, 85, 128], min: -0.070175, max: 0.766517, mean: 0.313738
        // tensor [x_3] size: [1, 3, 341, 512], min: -35.474194, max: 30.698671, mean: 0.738294
        // tensor [x_4] size: [1, 3, 341, 512], min: 0.0, max: 0.893776, mean: 0.316156

        return y;
    }
};

int autops_predict(int device, int n, char *input_files[], char *output_dir);

#endif // __DEEPGUIDEDFILTER__H__

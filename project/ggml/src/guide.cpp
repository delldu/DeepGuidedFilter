/************************************************************************************
***
*** Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "guide.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include <sys/stat.h> // for chmod

static ggml_tensor_t *ggml_nn_rows(struct ggml_context* ctx, ggml_tensor_t *x, int r);
static ggml_tensor_t *ggml_nn_cols(struct ggml_context* ctx, ggml_tensor_t *x, int r);

static ggml_tensor_t *ggml_diff_rows(struct ggml_context* ctx, ggml_tensor_t *x, int r)
{
    int W = (int)x->ne[0];
    int H = (int)x->ne[1];
    int C = (int)x->ne[2];
    int B = (int)x->ne[3];

    ggml_tensor_t* B1 = ggml_nn_slice(ctx, x, 1/*dim -- H*/, r, 2*r + 1, 1/*step*/);
    ggml_tensor_t* B21 = ggml_nn_slice(ctx, x, 1/*dim -- H*/, 2*r+1, H, 1/*step*/);
    ggml_tensor_t* B22 = ggml_nn_slice(ctx, x, 1/*dim -- H*/, 0, H - (2*r + 1), 1/*step*/);
    ggml_tensor_t* B2 = ggml_sub(ctx, B21, B22);

    ggml_tensor_t* B31 = ggml_nn_slice(ctx, x, 1/*dim -- H*/, H - r, H, 1/*step*/);
    ggml_tensor_t* B32 = ggml_nn_slice(ctx, x, 1/*dim -- H*/, H - (2*r + 1), H - (r + 1), 1/*step*/);
    ggml_tensor_t* B3 = ggml_sub(ctx, B31, B32);

    ggml_tensor_t* y = ggml_concat(ctx, B1, B2, 1/*dim--H*/);

    return ggml_concat(ctx, y, B3, 1/*dim--H*/);
}


static ggml_tensor_t *ggml_nn_cols(struct ggml_context* ctx, ggml_tensor_t *x, int r)
{
    int W = (int)x->ne[0];
    int H = (int)x->ne[1];
    int C = (int)x->ne[2];
    int B = (int)x->ne[3];

    ggml_tensor_t* B1 = ggml_nn_slice(ctx, x, 0/*dim -- W*/, r, 2*r + 1, 1/*step*/);
    ggml_tensor_t* B21 = ggml_nn_slice(ctx, x, 0/*dim -- W*/, 2*r+1, W, 1/*step*/);
    ggml_tensor_t* B22 = ggml_nn_slice(ctx, x, 0/*dim -- W*/, 0, W - (2*r + 1), 1/*step*/);
    ggml_tensor_t* B2 = ggml_sub(ctx, B21, B22);

    ggml_tensor_t* B31 = ggml_nn_slice(ctx, x, 0/*dim -- W*/, W - r, W, 1/*step*/);
    ggml_tensor_t* B32 = ggml_nn_slice(ctx, x, 0/*dim -- W*/, W - (2*r + 1), W - (r + 1), 1/*step*/);
    ggml_tensor_t* B3 = ggml_sub(ctx, B31, B32);

    ggml_tensor_t* y = ggml_concat(ctx, B1, B2, 0/*dim--W*/);

    return ggml_concat(ctx, y, B3, 0/*dim--W*/);
}

ggml_tensor_t *ggml_box_filter(struct ggml_context* ctx, ggml_tensor_t *x, int r)
{
    // ggml_tensor_dump("box_filter1", x);
    x = ggml_cumsum(ctx, x, 1/*dim*/); // cum sum on H
    // ggml_tensor_dump("box_filter2", x);
    x = ggml_diff_rows(ctx, x, r);
    // ggml_tensor_dump("box_filter3", x);

    x = ggml_cumsum(ctx, x, 0/*dim*/); // cum sum on W
    // ggml_tensor_dump("box_filter4", x);
    x = ggml_nn_cols(ctx, x, r);
    // ggml_tensor_dump("box_filter5", x);

    return x;
}


int autops_predict(int device, int n, char *input_files[], char *output_dir)
{
    struct DeepGuidedFilter net;
    char *p, output_fname[512];
    TENSOR *input_tensor, *output_tensor, *tensor_argv[1] = {};

    if (n < 1)
        return RET_OK;

    make_dir(output_dir);

    // load net weight ...
    GGMLModel model;
    {
        check_point(model.preload("models/image_autops.gguf") == RET_OK);
        net.set_device(device);
        net.start_engine();
        net.dump();
    }

    for (int i = 0; i < n; i++) {
        p = strrchr(input_files[i], '/');
        p = (!p) ? input_files[i] : p + 1;
        snprintf(output_fname, sizeof(output_fname) - 1, "%s/%s", output_dir, p);

        syslog_info("Autops %s to %s ...", input_files[i], output_fname);


        input_tensor = tensor_load_image(input_files[i], 0 /*without alpha*/);
        check_tensor(input_tensor);

        net.load_weight(&model, "");

        tensor_argv[0] = input_tensor;
        output_tensor = net.engine_forward(ARRAY_SIZE(tensor_argv), tensor_argv);
        check_tensor(output_tensor);
        if (output_tensor) {
            tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_fname);
            tensor_destroy(output_tensor);
        }

        TENSOR *xxxx_test = net.get_output_tensor("x_0");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_0", xxxx_test);
            tensor_destroy(xxxx_test);
        }
        xxxx_test = net.get_output_tensor("x_1");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_1", xxxx_test);
            tensor_destroy(xxxx_test);
        }
        xxxx_test = net.get_output_tensor("x_2");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_2", xxxx_test);
            tensor_destroy(xxxx_test);
        }
        xxxx_test = net.get_output_tensor("x_3");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_3", xxxx_test);
            tensor_destroy(xxxx_test);
        }
        xxxx_test = net.get_output_tensor("x_4");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** x_4", xxxx_test);
            tensor_destroy(xxxx_test);
        }

        xxxx_test = net.get_output_tensor("xxxx_test");
        if (tensor_valid(xxxx_test)) {
            tensor_show("********************** xxxx_test", xxxx_test);
            tensor_destroy(xxxx_test);
        }

        tensor_destroy(input_tensor);
    }

    // free network ...
    {
        model.clear();
        net.stop_engine();
    }

    return RET_OK;
}

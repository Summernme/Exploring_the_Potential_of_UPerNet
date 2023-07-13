#!/bin/bash
# MODEL_PATH=.ckpt/abcdefu-resnet50-upernet-ngpus1-batchSize4-imgMaxSize1000-paddingConst32-segmDownsampleRate4-LR_encoder0.02-LR_decoder0.02-epoch40-decay0.0001-fixBN0
MODEL_PATH=./ckpt/baseline-resnet50-upernet-ngpus2-batchSize8-imgMaxSize1000-paddingConst32-segmDownsampleRate4-LR_encoder0.02-LR_decoder0.02-epoch40-decay0.0001-fixBN0
TEST_IMG=ADE_val_00000001.jpg
BASE_RESULT_PATH=./ckpt/baseline-resnet50-upernet-ngpus2-batchSize8-imgMaxSize1000-paddingConst32-segmDownsampleRate4-LR_encoder0.02-LR_decoder0.02-epoch40-decay0.0001-fixBN0/

for i in {10..1}
do
    #RESULT_PATH="${BASE_RESULT_PATH}epoch_${i}/"

    echo "epoch $i"
    #mkdir -p $RESULT_PATH
    # python -u test.py \
    # --model_path $MODEL_PATH \
    # --test_img $TEST_IMG \
    # --arch_encoder resnet50 \
    # --arch_decoder upernet \
    # --result $MODEL_PATH \
    # --suffix "_epoch_$i"\
    # --save_ori_img 0

    python3 eval_multipro.py \
        --arch_encoder resnet50 \
        --arch_decoder upernet \
        --suffix "_epoch_$i.pth" \
        --id  unfreeze-4head-part \
        --devices 0
done

    # python3 -u test.py \
    #     --model_path $MODEL_PATH \
    #     --test_img $TEST_IMG \
    #     --arch_encoder resnet50 \
    #     --arch_decoder upernet \
    #     --result ./ \
    #     --suffix _epoch_37.pth \
    #     --gpu_id 0

    # python3 eval_multipro.py \
    #     --arch_encoder resnet50 \
    #     --arch_decoder upernet \
    #     --suffix _epoch_1.pth \
    #     --id valtest \
    #     --devices 0

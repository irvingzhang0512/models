export CUDA_VISIBLE_DEVICES=7

# Prerequisite
# 1. Make sure current python package support TensorFlow Object Detection API 1.x and `tflite_convert`
# 2. `MNN` is required if we want to convert .tflite to .mnn

# Features
# 1. Download ssd compressed files(.tar.gz) from tensorflow object detection api model zoo and uncompress them.
# 2. Convert downloaded ckpt(model.ckpt) to tflite format.
# 3. Convert tflite format to mnn format

OBJECT_DETECTION_PATH=$(pwd)
RESULT_FILE=${OBJECT_DETECTION_PATH}/result.txt

# Root of MNN, make sure `${MNN_PATH}/build/MNNConvert` exists
MNN_PATH=$HOME/mnn
# Directory to save .mnn files
MNN_CHECPOINT_PATH=/ssd/zhangyiyang/flashai/data/mnn

# Directory to save .tflite files
TFLITE_SAVE_DIR=/ssd/zhangyiyang/flashai/data/tflite

CHECKPOINT_PATH=${OBJECT_DETECTION_PATH}/checkpoints
if [ ! -d "${CHECKPOINT_PATH}" ]; then
mkdir ${CHECKPOINT_PATH}
fi

convert() {
    wget_url=$1
    shape=$2
    inference_type=$3
    
    checkpoint_sub=.
    if [ -n "$4" ] ;then
        checkpoint_sub=$4
    fi
    
    checkpoint_name=model.ckpt
    if [ -n "$5" ] ;then
        checkpoint_name=$5
    fi

    # downloaded ckpt
    # e.g. ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    tar_name=${wget_url##*/}

    # e.g. ssd_mobilenet_v1_coco_2018_01_28
    cur_ckpt_dir_name=${tar_name%%.tar.gz*}

    cur_tar_path=${CHECKPOINT_PATH}/${tar_name}
    cur_ckpt_dir_path=${CHECKPOINT_PATH}/${cur_ckpt_dir_name}

    echo "Start to convert ${cur_ckpt_dir_name}"

    # Step 1: download ckpt zip file and extract
    if [ ! -d "${cur_ckpt_dir_path}" ]; then
        if [ ! -f "$cur_tar_path" ]; then
            echo "Start to download ${tar_name}"
            wget -P $CHECKPOINT_PATH $wget_url
        else
            echo "No need to download {cur_tar_path} since it already exists."
        fi
        tar zxvf $cur_tar_path -C ${CHECKPOINT_PATH}
    else
        echo "${cur_ckpt_dir_path} already exists, no need to download and untar."
    fi

    if [ ! -d "${cur_ckpt_dir_path}" ]; then
        echo "Error Generate ${cur_ckpt_dir_path}" >> ${RESULT_FILE}
        return
    fi

    # # Step 2: convert model.ckpt -> tflite.pb
    # cur_ckpt_dir_path=${cur_ckpt_dir_path}/${checkpoint_sub}
    # cur_pb_path=${cur_ckpt_dir_path}/tflite_graph.pb
    # if [ ! -f "$cur_pb_path" ]; then
    #     echo "Start to convert model.ckpt to tflite_graph.pb"
    #     cd ${OBJECT_DETECTION_PATH}/..

    #     # # remove `batch_norm_trainable: true` from ssd mobilenet v2 pipeline 
    #     # sed -i '/batch_norm_trainable: true/d' ${cur_ckpt_dir_path}/pipeline.config

    #     python object_detection/export_tflite_ssd_graph.py \
    #         --pipeline_config_path ${cur_ckpt_dir_path}/pipeline.config \
    #         --trained_checkpoint_prefix ${cur_ckpt_dir_path}/${checkpoint_name} \
    #         --output_directory ${cur_ckpt_dir_path}
    # else
    #     echo "No need to convert ${cur_pb_path} since it already exists."
    # fi

    # if [ ! -f "$cur_pb_path" ]; then
    #     echo "Error Generate ${cur_pb_path}" >> ${RESULT_FILE}
    #     return
    # fi

    # Step 2: convert model.ckpt -> tflite.pb -> detect.tflite
    tflite_path=${cur_ckpt_dir_path}/detect.tflite
    if [ ! -f "$tflite_path" ]; then
        cur_ckpt_dir_path=${cur_ckpt_dir_path}/${checkpoint_sub}
        cur_pb_path=${cur_ckpt_dir_path}/tflite_graph.pb
        cd ${OBJECT_DETECTION_PATH}/..

        # remove `batch_norm_trainable: true` from ssd mobilenet v2 pipeline 
        sed -i '/batch_norm_trainable: true/d' ${cur_ckpt_dir_path}/pipeline.config

        # convert model.ckpt -> tflite.pb
        echo "Start to convert model.ckpt to tflite_graph.pb"
        python object_detection/export_tflite_ssd_graph.py \
            --pipeline_config_path ${cur_ckpt_dir_path}/pipeline.config \
            --trained_checkpoint_prefix ${cur_ckpt_dir_path}/${checkpoint_name} \
            --output_directory ${cur_ckpt_dir_path}

        if [ ! -f "$cur_pb_path" ]; then
            echo "Error Generate ${cur_pb_path}" >> ${RESULT_FILE}
            return
        fi

        echo "Start to convert tflite.pb to detect.tflite"
        tflite_convert \
            --inference_type=${inference_type} \
            --enable_v1_converter \
            --graph_def_file=${cur_ckpt_dir_path}/tflite_graph.pb \
            --output_file=${tflite_path} \
            --input_shapes=1,${shape},3 \
            --input_arrays=normalized_input_image_tensor \
            --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
            --mean_values=128 \
            --std_dev_values=128 \
            --change_concat_input_ranges=false \
            --allow_custom_ops
    else
        echo "$tflite_path already exists, no need to convert from model.ckpt to detect.tflite"
    fi

    if [ ! -f "${tflite_path}" ]; then
        echo "Error Generate ${tflite_path}" >> ${RESULT_FILE}
        return
    else
        if [ ! -d "${TFLITE_SAVE_DIR}" ]; then
            echo "Saving .tflite to ${TFLITE_SAVE_DIR}"
            cp $tflite_path ${TFLITE_SAVE_DIR}/${cur_ckpt_dir_name}.tflite
        fi
    fi
    
    # Step 4: convert .tflite to .mnn
    if [ -d "${MNN_PATH}" ]; then
        mnn_path=${MNN_CHECPOINT_PATH}/${cur_ckpt_dir_name}.mnn
        if [ ! -f "$mnn_path" ]; then
            echo "Start to convert ${tflite_path} to ${mnn_path}"
            ${MNN_PATH}/build/MNNConvert -f TFLITE --modelFile ${tflite_path} --MNNModel ${mnn_path} --bizCode biz
        else
            echo "No need to generate ${mnn_path} since it already exists"
        fi

        if [ ! -f "${mnn_path}" ]; then
            echo "Error Generate ${mnn_path}" >> ${RESULT_FILE}
        else
            echo "Successfully generate ${mnn_path}" >> ${RESULT_FILE}
        fi
    fi
}

# COCO-trained models
# mobilenet_v1
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz 300,300 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz 300,300 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz 300,300 QUANTIZED_UINT8
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz 300,300 QUANTIZED_UINT8
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz 300,300 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz 640,640 FLOAT
# ssd_resnet50_v1
convert http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz 640,640 FLOAT
# ssd_mobilenet_v2 & ssdlite_mobilenet_v2
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz 300,300 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz 300,300 QUANTIZED_UINT8
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz 300,300 FLOAT

# Mobile models
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tar.gz 320,320 FLOAT . model.ckpt-400000
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tar.gz 320,320 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz 320,320 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2020_01_14.tar.gz 320,320 FLOAT

# Pixel4 Edge TPU models
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz 320,320 FLOAT fp32 model.ckpt-400000
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz 320,320 QUANTIZED_UINT8 uint8 model.ckpt-400000
convert https://storage.cloud.google.com/mobilenet_edgetpu/checkpoints/ssdlite_mobilenet_edgetpu_coco_quant.tar.gz 320,320 FLOAT

# Pixel4 DSP models
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tar.gz 320,320 FLOAT fp32 model.ckpt-400000
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tar.gz 320,320 QUANTIZED_UINT8 uint8 model.ckpt-400000

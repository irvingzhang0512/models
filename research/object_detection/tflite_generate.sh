export CUDA_VISIBLE_DEVICES=7

OBJECT_DETECTION_PATH=$(pwd)
MNN_PATH=$HOME/mnn
RESULT_FILE=${OBJECT_DETECTION_PATH}/result.txt
MNN_CHECPOINT_PATH=/ssd/zhangyiyang/flashai/data/mnn
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

    tar_name=${wget_url##*/}
    cur_ckpt_dir_name=${tar_name%%.tar.gz*}
    cur_tar_path=${CHECKPOINT_PATH}/${tar_name}
    cur_ckpt_dir_path=${CHECKPOINT_PATH}/${cur_ckpt_dir_name}

    echo ""
    echo ""
    echo ""
    echo ""
    echo "Start to convert ${cur_ckpt_dir_name}"

    if [ ! -d "${cur_ckpt_dir_path}" ]; then
        if [ ! -f "$cur_tar_path" ]; then
            echo "Start to download ${tar_name}"
            wget -P $CHECKPOINT_PATH $wget_url
        fi
        echo "Start to untar"
        tar zxvf $cur_tar_path -C ${CHECKPOINT_PATH}
    else
        echo "${cur_ckpt_dir_path} already exists, no need to download and untar."
    fi

    if [ ! -d "${cur_ckpt_dir_path}" ]; then
        echo "Error Generate ${cur_ckpt_dir_path}" >> ${RESULT_FILE}
        return
    fi

    cur_ckpt_dir_path=${cur_ckpt_dir_path}/${checkpoint_sub}
    tflite_path=${cur_ckpt_dir_path}/detect.tflite
    if [ ! -f "$tflite_path" ]; then
        echo "Start to convert model.cpt to tflite.pb"
        cd ${OBJECT_DETECTION_PATH}/..
        python object_detection/export_tflite_ssd_graph.py \
            --pipeline_config_path ${cur_ckpt_dir_path}/pipeline.config \
            --trained_checkpoint_prefix ${cur_ckpt_dir_path}/${checkpoint_name} \
            --output_directory ${cur_ckpt_dir_path}

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
        cp $tflite_path ${TFLITE_SAVE_DIR}/${cur_ckpt_dir_name}.tflite
    fi
    
    if [ ! -f "$mnn_path" ]; then
        echo "Start to convert detection.tflite to ${mnn_path}"
        mnn_path=${cur_ckpt_dir_name}.mnn
        cd ${MNN_PATH}/build
        ./MNNConvert -f TFLITE --modelFile ${tflite_path} --MNNModel ${mnn_path} --bizCode biz
    fi

    if [ ! -f "${tflite_path}" ]; then
        echo "Error Generate ${mnn_path}" >> ${RESULT_FILE}
    else
        echo "Successfully generate ${mnn_path}" >> ${RESULT_FILE}
    fi
}

convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz 300,300 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz 300,300 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz 300,300 QUANTIZED_UINT8
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz 300,300 QUANTIZED_UINT8
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz 300,300 FLOAT

# convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz 640,640 FLOAT
# convert http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz 640,640 FLOAT

# ssd mobilenet v2 需要删除 pipeline 中的 batch_norm_trainable: true
convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz 300,300 FLOAT


convert http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz 300,300 QUANTIZED_UINT8
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz 300,300 FLOAT

convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz 320,320 FLOAT fp32 model.ckpt-400000
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tar.gz 320,320 QUANTIZED_UINT8 uint8 model.ckpt-400000
convert https://storage.cloud.google.com/mobilenet_edgetpu/checkpoints/ssdlite_mobilenet_edgetpu_coco_quant.tar.gz 320,320 FLOAT
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tar.gz 320,320 FLOAT fp32 model.ckpt-400000
convert http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tar.gz 320,320 QUANTIZED_UINT8 uint8 model.ckpt-400000

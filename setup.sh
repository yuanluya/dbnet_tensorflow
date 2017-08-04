#!/bin/sh
# ===========================
# Usage: ./setup.sh (model|data)?

if wget --help | grep -q 'show-progress'; then
    WGET_FLAG="-q --show-progress"
else
    WGET_FLAG=""
fi

# create a tmp directory for the downloading data
TMP_DIR="./tmp_download"
mkdir -p "${TMP_DIR}"

# downloading model
download_model() 
{
    # directory for model
    MODEL_TAR_BALL="${TMP_DIR}/pretrained_model.tar.gz"
    MODEL_DIR="${TMP_DIR}/pretrained_model"
    mkdir -p "${MODEL_DIR}"

    MODEL_URL="http://www.ytzhang.net/files/dbnet/tensorflow/dbnet-vgg-pretrained.tar.gz"
    echo "Downloading pre-trained models ..."
    wget ${WGET_FLAG} "${MODEL_URL}" -O "${MODEL_TAR_BALL}"
    echo "Uncompressing pre-trained models ..."
    tar -xzf "${MODEL_TAR_BALL}" -C "${MODEL_DIR}"

    # move model to default directories
    VGG_REGION_NET_DIR="./networks/image_feat_net/vgg16"
    RESNET_REGION_NET_DIR="./networks/image_feat_net/resnet101"
    TEXT_NET_DIR="./networks/text_feat_net"
    echo "Move pre-trained image network model to ${VGG_REGION_NET_DIR} ..."
    mv ${MODEL_DIR}/vgg16_Region_Feat_Net.npy "${VGG_REGION_NET_DIR}/Region_Feat_Net.npy"
    mv ${MODEL_DIR}/vgg16_frcnn_Region_Feat_Net.npy "${VGG_REGION_NET_DIR}/frcnn_Region_Feat_Net.npy"
    echo "Move pre-trained image network model to ${RESNET_REGION_NET_DIR} ..."
    mv ${MODEL_DIR}/resnet101_Region_Feat_Net.npy "${RESNET_REGION_NET_DIR}/Region_Feat_Net.npy"
    mv ${MODEL_DIR}/resnet101_frcnn_Region_Feat_Net.npy "${RESNET_REGION_NET_DIR}/frcnn_Region_Feat_Net.npy"
    echo "Move pre-trained text network model to ${TEXT_NET_DIR} ..."
    mv ${MODEL_DIR}/*Text*.npy "${TEXT_NET_DIR}"
}

# downloading data
download_data() 
{
    # directory for data
    DATA_TAR_BALL="${TMP_DIR}/data.tar.gz"
    DATA_DIR="./data"
    mkdir -p "${DATA_DIR}"

    DATA_URL="http://www.ytzhang.net/files/dbnet/data/vg_v1_json_.tar.gz"
    echo "Downloading data ..."
    wget ${WGET_FLAG} "${DATA_URL}" -O "${DATA_TAR_BALL}"
    echo "Uncompressing data ..."
    tar -xzf "${DATA_TAR_BALL}" -C "${DATA_DIR}"
}

# default to download all
if [ $# -eq 0 ]; then
    download_model
    download_data
else
    case $1 in
        "model") download_model
            ;;
        "data") download_data
            ;;
        *) echo "Usage: ./setup.sh [OPTION]"
           echo ""
           echo "No option will download both model and data."
           echo ""
           echo "OPTION:\n\tmodel: only download the pre-trained models (.npy)"
           echo "\tdata: only download the data(.json)"
            ;;
    esac
fi

# clear the tmp files
rm -rf "${TMP_DIR}"

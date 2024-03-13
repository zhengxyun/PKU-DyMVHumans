#!/usr/bin/env bash

gpu=2
dataset_nm=4K_Studios_Show_Pair_f16f17
database_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1/${dataset_nm}/per_frame/000000
DATASET_PATH=${database_path}/run_colmap
mkdir ${DATASET_PATH}
cp -r ${database_path}/images ${DATASET_PATH}/
CAMERA_MODEL=PINHOLE

# 1. colmap extric and intric parameters"

mkdir $DATASET_PATH/colmap

colmap feature_extractor \
  --database_path "$DATASET_PATH/colmap/database.db" \
  --image_path "$DATASET_PATH/images" \
  --ImageReader.camera_model "$CAMERA_MODEL"

colmap exhaustive_matcher \
  --database_path "$DATASET_PATH/colmap/database.db"

mkdir $DATASET_PATH/colmap/sparse

colmap mapper \
    --database_path "$DATASET_PATH/colmap/database.db" \
    --image_path "$DATASET_PATH/images" \
    --output_path "$DATASET_PATH/colmap/sparse"

colmap model_converter \
    --input_path "$DATASET_PATH/colmap/sparse/0/" \
    --output_path "$DATASET_PATH/colmap/sparse" \
    --output_type TXT

### optional
grep "$CAMERA_MODEL" $DATASET_PATH/colmap/sparse/cameras.txt > $DATASET_PATH/colmap/sparse/intrinsic.txt
grep  "png" $DATASET_PATH/colmap/sparse/images.txt   > $DATASET_PATH/colmap/sparse/extrinsic.txt

#sort intinsic.txt by camera id.
sort -n -k 1 -t ' '  $DATASET_PATH/colmap/sparse/intrinsic.txt -o  $DATASET_PATH/colmap/sparse/intrinsic-sort.txt
#sort extrinsic.txt by image id.
sort -n -k 10 -t ' '  $DATASET_PATH/colmap/sparse/extrinsic.txt -o  $DATASET_PATH/colmap/sparse/extrinsic-sort.txt

echo "finished extric and intric parameters"

# 2. post_colmap - dense

mkdir $DATASET_PATH/colmap/dense

colmap image_undistorter \
    --image_path "$DATASET_PATH/images" \
    --input_path "$DATASET_PATH/colmap/sparse/0" \
    --output_path "$DATASET_PATH/colmap/dense" \
    --output_type COLMAP \
    --max_image_size 2000

CUDA_VISIBLE_DEVICES=${gpu} colmap patch_match_stereo \
    --workspace_path "$DATASET_PATH/colmap/dense" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

#colmap stereo_fusion \
#    --workspace_path "$DATASET_PATH/colmap/dense" \
#    --workspace_format COLMAP \
#    --input_type geometric \
#    --output_path "$DATASET_PATH/colmap/dense/fused.ply"


# 3. colmap2mvs_acmp
# Use colmap2mvsnet_acm.py with default parameters to convert the COLMAP files to MVSNet format.
mkdir $DATASET_PATH/colmap2mvs_casNet

python scripts/colmap2mvsnet_acmp.py \
  --dense_folder="$DATASET_PATH/colmap/dense" \
  --save_folder="$DATASET_PATH/colmap2mvs_casNet" \
  --model_ext ".bin"

cp -r $DATASET_PATH/colmap2mvs_casNet/cams /amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1/${dataset_nm}/
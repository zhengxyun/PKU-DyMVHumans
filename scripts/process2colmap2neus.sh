# init data format for colmap2neus/neus2

# step1: Run COLMAP SfM
# After running the commands above, a sparse point cloud is saved in ${data_dir}/sparse_points.ply.
# copy data for data_COLMAP
gpu=2
rdata_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1
work_path=/amax/zxyun/FreeViewSynthesis/Mypapers/DyMVHumans
nm=(1080_Dance_Dunhuang_Single_f14 1080_Sport_Badminton_Single_f11 1080_Kungfu_Weapon_Pair_m12m13 4K_Studios_Show_Pair_f16f17 4K_Studios_Show_Groups
    1080_Dance_Dunhuang_Pair_f14f15 1080_Sport_Football_Single_m11 1080_Kungfu_Fan_Single_m12
    )

for (( v=1; v<2; v++)); do
    case_nm=${nm[v]}
    echo "This is ${case_nm}"
    data_path=${rdata_path}/${case_nm}/per_frame
    mkdir ${data_path}/tmp2mac
    # copy for data_COLMAP
    colmap_path=${rdata_path}/${case_nm}/data_COLMAP
    mkdir ${colmap_path}
    for k in $(seq -f "%06g" 0 5 30); do
#    for k in 000035 ; do
      echo "this is frame ${k}"
      CUDA_VISIBLE_DEVICES=${gpu} python ${work_path}/scripts/neus_data_process/imgs2poses.py ${data_path}/${k}
      mkdir ${data_path}/tmp2mac/${k}
      cp ${data_path}/${k}/sparse_points.ply ${data_path}/tmp2mac/${k}

      # copy for data_COLMAP
      mkdir ${colmap_path}/${k}
      cp -r ${data_path}/${k}/images ${colmap_path}/${k}
      cp -r ${data_path}/${k}/sparse ${colmap_path}/${k}
      cp ${data_path}/${k}/colmap_output.txt ${colmap_path}/${k}/
      cp -r ${data_path}/${k}/pha ${colmap_path}/${k}/masks
      cd ${colmap_path}/${k}/masks
      for file in `ls *.png`;do mv $file ${file:4:14}.png;done;
      for file in `ls *.png`;do mv $file image_${file};done;

    done;
    cd ${work_path}
done

# Step 2. Define the region of interest, you can refer to https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data
# then Data Convention for NeuS
rdata_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1
work_path=/amax/zxyun/FreeViewSynthesis/Mypapers/DyMVHumans
nm=(1080_Dance_Dunhuang_Single_f14 1080_Sport_Badminton_Single_f11 1080_Kungfu_Weapon_Pair_m12m13 4K_Studios_Show_Groups 4K_Studios_Show_Pair_f16f17
    1080_Dance_Dunhuang_Pair_f14f15 1080_Sport_Football_Single_m11 1080_Kungfu_Fan_Single_m12
    )
for (( v=1; v<2; v++)); do
    case_nm=${nm[v]}
    echo "This is ${case_nm}"
    data_path=${rdata_path}/${case_nm}/per_frame
    data_neus=${rdata_path}/${case_nm}/data_NeuS
    mkdir ${data_neus}
    for k in $(seq -f "%06g" 0 5 30); do
#    for k in 000035 ; do
      echo "this is frame ${k}"
      #copy
      cp ${data_path}/mac2tmp/${k}/sparse_points_interest.ply ${data_path}/${k}/sparse_points_interest.ply
      python ${work_path}/scripts/neus_data_process/gen_cameras.py ${data_path}/${k}
      # copy preprocessed
      mv ${data_path}/${k}/preprocessed ${data_neus}/${k}

      # remove files
      rm ${data_path}/${k}/database.db
      rm ${data_path}/${k}/poses.npy
      rm ${data_path}/${k}/pose.ply

      # rename mask
      rm -r ${data_neus}/${k}/mask
      cp -r ${data_path}/${k}/pha ${data_neus}/${k}/mask
      cd ${data_neus}/${k}/mask
      for file in `ls *.png`;do mv $file ${file:6:3}.png;done;
    done;
done

# Step 3. Data Convention for NeuS2
rdata_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1
work_path=/amax/zxyun/FreeViewSynthesis/Mypapers/DyMVHumans
nm=(1080_Dance_Dunhuang_Single_f14 1080_Sport_Badminton_Single_f11 1080_Kungfu_Weapon_Pair_m12m13 4K_Studios_Show_Groups 4K_Studios_Show_Pair_f16f17
    1080_Dance_Dunhuang_Pair_f14f15 1080_Sport_Football_Single_m11 1080_Kungfu_Fan_Single_m12
    )
for (( v=1; v<2; v++)); do
    case_nm=${nm[v]}
    echo "This is ${case_nm}"
    data_path=${rdata_path}/${case_nm}/per_frame
    data_neus2=${rdata_path}/${case_nm}/data_NeuS2
    mkdir ${data_neus2}
    mkdir ${data_neus2}/images
    for k in $(seq -f "%06g" 0 5 30); do
#    for k in 000035 ; do
      echo "this is frame ${k}"
      # copy com_rgba for images
      cp -r ${data_path}/${k}/com ${data_neus2}/images/${k}
      # Data Convention
      python ${work_path}/scripts/neus_to_neus2.py --base_par_dir ${rdata_path} --dataset_name ${case_nm} --dataset_fme ${k}
    done;
done

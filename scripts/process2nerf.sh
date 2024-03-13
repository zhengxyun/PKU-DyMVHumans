# colmap2nerf for instant-ngp/torch_ngp

gpu=1
rdata_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1
work_path=/amax/zxyun/FreeViewSynthesis/Mypapers/DyMVHumans
nm=(1080_Dance_Dunhuang_Single_f14 1080_Sport_Badminton_Single_f11 1080_Kungfu_Weapon_Pair_m12m13 4K_Studios_Show_Pair_f16f17 4K_Studios_Show_Groups
    1080_Dance_Dunhuang_Pair_f14f15 1080_Sport_Football_Single_m11 1080_Kungfu_Fan_Single_m12
    )
for (( v=1; v<6; v++)); do
    case_nm=${nm[v]}
    echo "This is ${case_nm}"
    data_ngp=${rdata_path}/${case_nm}/data_ngp
    img_path=${rdata_path}/${case_nm}/per_frame
    mkdir ${data_ngp}
    for i in $(seq -f "%06g" 0 5 30); do
      echo "this is frame ${i}"
      mkdir ${data_ngp}/${i}
      cp -r ${img_path}/${i}/images ${data_ngp}/${i}/image
      colmap_img=${data_ngp}/${i}/image
      CUDA_VISIBLE_DEVICES=${gpu} python ${work_path}/scripts/colmap2nerf_ngp.py --images ${colmap_img} --run_colmap
      rm ${data_ngp}/${i}/colmap.db

      # replace image_with_backdround with com_rgba
      mv ${data_ngp}/${i}/image ${data_ngp}/${i}/images
      cp -r ${img_path}/${i}/rgba ${data_ngp}/${i}/image

    done;
done

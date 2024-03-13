
# run the data preprocessing script.
# 1. image per view

# 1.0 per view image overview
rdata_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1
work_path=/amax/zxyun/FreeViewSynthesis/Mypapers/DyMVHumans
nm=(1080_Dance_Dunhuang_Single_f14 1080_Sport_Badminton_Single_f11 1080_Kungfu_Weapon_Pair_m12m13 4K_Studios_Show_Groups 4K_Studios_Show_Pair_f16f17
    1080_Dance_Dunhuang_Pair_f14f15 1080_Sport_Football_Single_m11 1080_Kungfu_Fan_Single_m12
    )
for (( v=1; v<9; v++)); do
   case_nm=${nm[v]}
   echo "This is ${case_nm}"
   for (( k=0; k<250; k+=50)); do
     echo "This is ${k}"
     python ${work_path}/scripts/multiview_merge.py \
        --data_folder ${rdata_path} \
        --scene_nm ${case_nm} \
        --fme_id ${k} \
        --arr_nm 8
   done;
done

# 1.1 videos to images
# save the image sequence and matting results
gpu=2
rdata_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1
work_path=/amax/zxyun/FreeViewSynthesis/Mypapers/DyMVHumans
nm=(1080_Dance_Dunhuang_Single_f14 1080_Sport_Badminton_Single_f11 1080_Kungfu_Weapon_Pair_m12m13 4K_Studios_Show_Groups 4K_Studios_Show_Pair_f16f17
    1080_Dance_Dunhuang_Pair_f14f15 1080_Sport_Football_Single_m11 1080_Kungfu_Fan_Single_m12
    )
for (( v=2; v<3; v++)); do
    case_nm=${nm[v]}
    echo "This is ${case_nm}"
    mkdir ${rdata_path}/${case_nm}/per_view
    cam_nm=60                                                             # camera number（56 or 60）
    for (( i=0; i<${cam_nm}; i++)); do
        VIDEO_PATH="${rdata_path}/${case_nm}/videos/${i}.mp4"             # input video path
        echo "This is camera ${i}"
        SAVE_PATH="${rdata_path}/${case_nm}/per_view/cam_${i}"            # output folder
        START_FRAME=0                                                     # start frame of the video
        END_FRAME=250                                                     # end frame of the video
        INTERVAL=1                                                        # sampling interval
        METHOD='bgmv2'                                                    # choose matting method (bgmv2 or rvm)
        BGR_PATH="${work_path}/dataset/background/1080"                     # background img （4K or 1080）
        CUDA_VISIBLE_DEVICES=${gpu} python ${work_path}/scripts/preprocess.py --input_video ${VIDEO_PATH} \
                            --output_folder ${SAVE_PATH} \
                            --matting_method ${METHOD} \
                            --bgr_path ${BGR_PATH} \
                            --start_frame ${START_FRAME} \
                            --end_frame ${END_FRAME} \
                            --cam_nm ${i} \
                            --interval ${INTERVAL}
    done;
done

# 1.2 image per frame

rdata_path=/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part1
work_path=/amax/zxyun/FreeViewSynthesis/Mypapers/DyMVHumans
nm=(1080_Dance_Dunhuang_Single_f14 1080_Sport_Badminton_Single_f11 1080_Kungfu_Weapon_Pair_m12m13 4K_Studios_Show_Groups 4K_Studios_Show_Pair_f16f17
    1080_Dance_Dunhuang_Pair_f14f15 1080_Sport_Football_Single_m11 1080_Kungfu_Fan_Single_m12
    )
for (( v=1; v<6; v++)); do
  case_nm=${nm[v]}
  echo "This is ${case_nm}"
  fme_end=31                 # end frame of the video
  fme_itr=5                  # sampling interval
  python ${work_path}/scripts/cam2frames.py \
      --data_folder ${rdata_path} \
      --scene_nm ${case_nm} \
      --fme_st 0 \
      --fme_end ${fme_end} \
      --fme_itr ${fme_itr} \
      --cam_st 0
done

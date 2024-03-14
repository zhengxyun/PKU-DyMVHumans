#!/usr/bin/env
# -*- coding: utf-8 -*-

import cv2
import os
import glob
from PIL import Image
import numpy as np
import argparse
from utils.common_utils import *

def cam2frames(args):

    data_path = args.data_folder
    img_path = os.path.join(data_path, args.scene_nm, "per_view")
    out_path = os.path.join(data_path,args.scene_nm, "per_frame")

    if not os.path.exists(out_path):
        os.makedirs(out_path) 

    for i in range(args.fme_st, args.fme_end+1, args.fme_itr):
        print("frame:", i)
        out_dir = os.path.join(out_path,"%06d" % (i))
        img_out_dir = os.path.join(out_dir, "images")
        # mask_out_dir = os.path.join(out_dir, "mask")
        pha_out_dir = os.path.join(out_dir, "pha")
        rgba_out_dir = os.path.join(out_dir, "rgba")

        # com with white or black bkgd
        com_rgb_dir = os.path.join(out_dir, "com_rgb")

        os.makedirs(img_out_dir, exist_ok=True)
        # os.makedirs(mask_out_dir, exist_ok=True)
        os.makedirs(pha_out_dir, exist_ok=True)
        os.makedirs(rgba_out_dir, exist_ok=True)
        os.makedirs(com_rgb_dir, exist_ok=True)

        if '4K' in args.scene_nm:
            cam_end = 56
        else:
            cam_end = 60

        for j in range(args.cam_st,cam_end):
            img_pth = os.path.join(img_path, "cam_%s" % (j), "images","%06d.png" %i)
            # mask_pth = os.path.join(img_path, "cam_%s" % (j), "mask", "%06d.png" % i)
            pha_pth = os.path.join(img_path, "cam_%s" % (j), "pha", "%06d.png" % i)
            rgba_pth = os.path.join(img_path, "cam_%s" % (j), "com", "%06d.png" % i)

            img = Image.open(img_pth)
            # mask = Image.open(mask_pth)
            pha = Image.open(pha_pth)
            rgba = Image.open(rgba_pth)

            if img is None:
                print("open error")
                exit()

            img_out_pth = os.path.join(img_out_dir,"image_c_%03d_f_%06d.png" %(j,i))
            # mask_out_pth = os.path.join(mask_out_dir, "mask_c_%03d_f_%06d.png" %(j,i))
            pha_out_pth = os.path.join(pha_out_dir, "pha_c_%03d_f_%06d.png" %(j,i))
            rgba_out_pth = os.path.join(rgba_out_dir, "image_c_%03d_f_%06d.png" % (j, i))
            com_rgb_pth = os.path.join(com_rgb_dir, "image_c_%03d_f_%06d.png" %(j,i))

            img.save(img_out_pth)
            # mask.save(mask_out_pth)
            pha.save(pha_out_pth)
            rgba.save(rgba_out_pth)


            # 指定背景色:
            # 黑色(0,0,0) 白色(255, 255, 255) 绿幕(120, 255, 155)
            if '4K' in args.scene_nm:
                img_size = (3840, 2160)
            else:
                img_size = (1920, 1080)
            bgr = Image.new('RGB', size=img_size, color=(255, 255, 255))
            bgr_np = pil_to_np(bgr)
            # print("bgr_np:",bgr_np.shape)
            img_np = pil_to_np(img)
            pha_np = pil_to_np(pha)

            # print("bgr shape:", bgr_np.shape)
            # print("pha shape:", pha_np.shape)

            com_rgb = img_np * pha_np + bgr_np * (1 - pha_np)
            com_rgb_pil = np_to_pil(com_rgb)

            com_rgb_pil.save(os.path.join(com_rgb_pth))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rename file')

    parser.add_argument('--data_folder', required=True, type=str, help='source folder.')
    parser.add_argument('--scene_nm', required=True, type=str, help='scene name')
    parser.add_argument('--fme_st', type=int, default=0, help='start frame')
    parser.add_argument('--fme_end', type=int, default=299, help='end frame')
    parser.add_argument('--fme_itr', type=int, default=5, help='start frame')
    parser.add_argument('--cam_st', type=int, default=0, help='start cam')

    args = parser.parse_args()

    cam2frames(args)




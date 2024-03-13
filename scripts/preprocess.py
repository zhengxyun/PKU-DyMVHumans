#!/usr/bin/env
# -*- coding: utf-8 -*-

import sys

sys.path.append("..")
import os
import os.path as osp
import argparse
import imageio
import numpy as np
from tqdm import tqdm
from utils.common_utils import clip_video, matting, find_max_region
from glob import glob

args = argparse.ArgumentParser(description='Clip the input video and save to iamge squence.')
args.add_argument('--input_video', required=True, type=str, help='input video path.')
args.add_argument('--output_folder', required=True, type=str, help='input video path.')
args.add_argument('--start_frame', default=0, type=int, help='clip start frame index.')
args.add_argument('--end_frame', default=-1, type=int, help='clip end frame index.')
args.add_argument('--interval', default=1, type=int, help='sampling interval of the video.')
args.add_argument('--cam_nm', default=0, type=int, help='sampling interval of the video.')
args.add_argument('--thres', default=128, type=int, help='threshold for matting.')
args.add_argument('--matting_method', default='rvm', type=str, help='choose matting method.')
args.add_argument('--bgmv2_path', default='./BackgroundMattingV2', type=str, help='bgmv2 code path for foreground matting.')
args.add_argument('--bgr_path', default='1080', type=str, help='background img  (4K or 1080) for BackgroundMattingV2.')

args = args.parse_args()


def main(args):
    # prepare save folder
    os.makedirs(args.output_folder + '/images', exist_ok=True)
    # os.makedirs(args.output_folder + '/images', exist_ok=True)

    # clip the video
    video_path = args.input_video
    assert osp.exists(video_path), '{} does not exist'.format(video_path)
    start_frame, end_frame, interval = args.start_frame, args.end_frame, args.interval
    if end_frame == -1:
        end_frame = int(imageio.get_reader(video_path).count_frames())
    print('Clip video {} from {} to {} with interval {}'.format(video_path, start_frame, end_frame, interval))

    # save the image sequence
    frames = clip_video(video_path, start_frame, end_frame, interval)
    # video_path = args.input_video.replace('.mp4', f'_clip_from{start_frame}to{end_frame}with_interval{interval}.mp4')
    video_path = args.output_folder + '/video_clip.mp4'
    for i, frame in tqdm(enumerate(frames)):
        imageio.imwrite(args.output_folder + '/images/{:06d}.png'.format(i), frame)
    imageio.mimsave(video_path, frames, fps=25)

    # matting
    if args.matting_method=='rvm':
        matting(args.output_folder + '/images', args.output_folder, thres=args.thres)
    else:
        print("matting with bgmv2")
        # run bgmv2 to get matting results
        bgr_path = args.bgr_path + '/{}.png'.format(args.cam_nm)
        print("bgr_path:", bgr_path)
        out_path=args.output_folder
        cmd = f'cd {args.bgmv2_path} && \
        python inference_video.py --video-src {video_path} --video-bgr {bgr_path} --output-dir {out_path} --output-type com pha '# && \
        # cp {out_path}/com/* {out_path}/
        os.system(cmd)

        # post-processing
        thres=196
        os.makedirs(f'{out_path}/mask', exist_ok=True)
        # os.makedirs(f'{out_path}/com-mask', exist_ok=True)
        print(f'post-processing: {out_path}')
        for i in tqdm(range(len(glob(osp.join(out_path, 'com/*.png'))))):
            alpha = imageio.imread(f'{out_path}/pha/{i:06d}.png')
            alpha[alpha < thres] = 0
            alpha[alpha >= thres] = 255
            alpha = find_max_region(alpha)
            imageio.imwrite(f'{out_path}/mask/{i:06d}.png', alpha)



if __name__ == '__main__':
    main(args)

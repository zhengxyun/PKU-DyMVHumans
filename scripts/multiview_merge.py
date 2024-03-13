#!/usr/bin/env
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse

def mv_merge(args):

    data_path = os.path.join(args.data_folder, args.scene_nm)
    folder = os.path.join(data_path, "videos")
    ls = os.listdir(folder)
    # print(ls)


    pick = args.fme_id  # frame id to pick
    n = args.arr_nm  # array number

    patch_list = []

    # sort videos
    tokens = []
    for pa in ls:
        tokens.append(int(pa.split(".")[0]))

    for token in sorted(tokens):
        vna = os.path.join(folder, str(token) + '.mp4')
        reader = cv2.VideoCapture(vna)
        more_frame = True
        c = 0
        while more_frame:
            more_frame, frame = reader.read()
            h, w, _ = frame.shape
            if c == pick:
                frame = cv2.resize(frame, (w // n * 2, h // n * 2))
                hs, ws, _ = frame.shape
                cv2.putText(frame, str(token), (ws // 8, hs // 8), cv2.FONT_HERSHEY_SIMPLEX, 5 / n, (0, 0, 255), 2)
                patch_list.append(frame)
                # cv2.imwrite("tmp.png", frame)
                break
            c += 1
        reader.release()
    # print(len(patch_list))
    black = np.zeros_like(patch_list[0])
    if len(patch_list) % n != 0:
        for _ in range(n - len(patch_list) % n):
            patch_list.append(black)
    # print(len(patch_list))

    rows = []
    for tk in range(0, len(patch_list), n):
        rows.append(np.hstack(patch_list[tk:tk + n]))
    merge = np.vstack(rows)
    cv2.imwrite(os.path.join(data_path, "overview_fme_%s.png" % pick), merge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rename file')

    parser.add_argument('--data_folder', required=True, type=str, help='source folder.')
    parser.add_argument('--scene_nm', required=True, type=str, help='scene name')
    parser.add_argument('--fme_id', type=int, default=0, help='pick frame id')
    parser.add_argument('--arr_nm', type=int, default=6, help='array number')

    args = parser.parse_args()

    mv_merge(args)
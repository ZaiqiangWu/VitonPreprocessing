import argparse

import cv2
import numpy as np
from vid2agnostic import vid2agnostic_fullbody
from vid2densepose import vid2densepose
from vid2crop43 import vid2crop43
import os


def main(input_video,target_dir,name):
    dp_video_path = os.path.join(target_dir,'densepose',name+'.mp4')
    masked_video_path = os.path.join(target_dir, 'agnostic', name + '.mp4')
    mask_video_path = os.path.join(target_dir, 'agnostic_mask', name + '.mp4')
    video_path = os.path.join(target_dir, "videos",name + '.mp4')
    os.makedirs(os.path.join(target_dir,'densepose'),exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'agnostic'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'agnostic_mask'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'videos'), exist_ok=True)

    vid2agnostic_fullbody(input_video,masked_video_path,mask_video_path)
    vid2densepose(input_video, dp_video_path)
    vid2crop43(input_video,video_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video", type=str
    )
    parser.add_argument(
        "-n", "--name", type=str
    )
    parser.add_argument(
        "-t", "--target", type=str
    )
    args = parser.parse_args()

    main(args.input_video, args.target, args.name)
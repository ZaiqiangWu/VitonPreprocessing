import argparse

import cv2
import numpy as np
from model.DensePose.densepose_extractor import DensePoseExtractor
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from tqdm import tqdm
from util.image_warp import crop2_43
from PIL import Image



def vid2crop43(input_video_path="./input_video.mp4", output_video_path="./output_video.mp4"):
    video_loader = MultithreadVideoLoader(input_video_path)
    video_writer = Image2VideoWriter()
    densepose_extractor = DensePoseExtractor()
    for i in tqdm(range(len(video_loader))):

        frame = video_loader.cap()

        out_frame= crop2_43(frame)
        video_writer.append(out_frame)
    video_writer.make_video(outvid=output_video_path,fps=video_loader.fps_list[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video_path", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-o", "--output_video_path", type=str, default="./output_video.mp4"
    )
    args = parser.parse_args()

    vid2crop43(args.input_video_path, args.output_video_path)
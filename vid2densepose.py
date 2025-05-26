import argparse

import cv2
import numpy as np
from model.DensePose.densepose_extractor import DensePoseExtractor
from util.multithread_video_loader import MultithreadVideoLoader
#from util.image2video import Image2VideoWriter
from util.multithread_video_writer import MultithreadVideoWriter
from tqdm import tqdm
from util.image_warp import crop2_43
from PIL import Image



def vid2densepose(input_video_path="./input_video.mp4", output_video_path="./output_video.mp4"):
    video_loader = MultithreadVideoLoader(input_video_path)
    video_writer = MultithreadVideoWriter(output_video_path,fps=video_loader.get_fps())
    densepose_extractor = DensePoseExtractor()
    for i in tqdm(range(len(video_loader))):

        frame = video_loader.cap()

        frame=crop2_43(frame)
        out_frame = densepose_extractor.get_dp_map(frame,isRGB=False)
        video_writer.append(out_frame)
    video_writer.make_video()
    video_writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video_path", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-o", "--output_video_path", type=str, default="./output_video.mp4"
    )
    args = parser.parse_args()

    vid2densepose(args.input_video_path, args.output_video_path)
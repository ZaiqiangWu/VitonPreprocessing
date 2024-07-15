import argparse

import cv2
import numpy as np
from model.humanparsing.run_parsing import Parsing
from model.openpose.run_openpose import OpenPose

from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from tqdm import tqdm
from PIL import Image
from util.utils_ootd import get_mask_location
from util.image_warp import crop2_43


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

model_type = "dc" # "hd" or "dc" dc->full body



def vid2agnostic(input_video_path="./input_video.mp4", output_masked_path="./output_video.mp4",output_mask_path="./output_video.mp4"):
    video_loader = MultithreadVideoLoader(input_video_path)
    video_writer = Image2VideoWriter()
    mask_video_writer = Image2VideoWriter()
    parsing_model = Parsing(gpu_id=0)
    openpose_model = OpenPose(gpu_id=0)

    for i in tqdm(range(len(video_loader))):
        frame = video_loader.cap()
        frame = crop2_43(frame)
        h,w = frame.shape[:2]
        frame = frame[:, :, [2, 1, 0]]
        frame = Image.fromarray(frame)

        model_img = frame.resize((768, 1024))
        model_parse, _ = parsing_model(model_img.resize((384, 512)))
        keypoints = openpose_model(model_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, 'upper_body', model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_img, mask)
        raw_result = masked_vton_img.resize((w,h), Image.NEAREST)
        raw_mask = mask.resize((w,h), Image.NEAREST)
        raw_mask=np.array(raw_mask)
        raw_mask=raw_mask[:,:,np.newaxis]
        raw_mask=raw_mask[:,:,[0,0,0]]
        out_frame = raw_result[:,:,[2,1,0]]

        video_writer.append(out_frame)
        mask_video_writer.append(raw_mask)
    video_writer.make_video(outvid=output_masked_path,fps=video_loader.fps_list[0])
    mask_video_writer.make_video(outvid=output_mask_path,fps=video_loader.fps_list[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_video_path", type=str, default="./input_video.mp4"
    )
    parser.add_argument(
        "-o", "--output_video_path", type=str, default="./output_video.mp4"
    )
    args = parser.parse_args()

    vid2agnostic(args.input_video_path, args.output_video_path)
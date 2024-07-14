import argparse

import cv2
import numpy as np
from model.humanparsing.run_parsing import Parsing
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from tqdm import tqdm
from PIL import Image



def main(input_video_path="./input_video.mp4", output_video_path="./output_video.mp4"):
    video_loader = MultithreadVideoLoader(input_video_path)
    video_writer = Image2VideoWriter()
    parsing_model = Parsing(gpu_id=0)
    for i in tqdm(range(len(video_loader))):
        frame = video_loader.cap()
        frame = frame[:, :, [2, 1, 0]]
        frame = Image.fromarray(frame)
        img_reshaper = ImageReshaper(frame)
        frame_43 = img_reshaper.get_reshaped()
        model_img = frame_43.resize((768, 1024))
        model_parse, _ = parsing_model(model_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, model_img, mask)
        out_frame = densepose_extractor.get_dp_map(frame,isRGB=False)
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

    main(args.input_video_path, args.output_video_path)
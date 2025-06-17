# VitonPreprocessing

Preprocess script for ViViD

## How to use

```

conda create --name vip python=3.9
conda activate vip

pip install -r requirements.txt
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git


python preprocess.py --input_video input_video.mp4 --name video_name --target target_path
```
import argparse
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

import options as option
from models import create_model
sys.path.insert(0, "../../")
from img_utils import tensor2img
import gradio as gr
import os
from PIL import Image


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt", type=str, default="test_setting2.yml", help="Path to options YMAL file."
)
parser.add_argument("-input_dir", type=str, default="../../../data_samples/")
parser.add_argument("-output_dir", type=str, default="../../../data_samples/")
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)
model = create_model(opt)


if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)


def predict(img):
    # img = cv2.imread(args.input_dir)[:, :, [2, 1, 0]]
    img = img.transpose(2, 0, 1)[None] / 255
    img_t = torch.as_tensor(np.ascontiguousarray(img)).float()
    model.feed_data(img_t)
    model.test()
    sr = model.fake_SR.detach().float().cpu()[0]
    sr_im = tensor2img(sr)
    sr_im = sr_im[:, :, ::-1]
    return sr_im


i = gr.inputs.Image()
o = gr.outputs.Image()

title = "DAN: Unfolding the Alternating Optimization for Blind Super Resolution"
description = "A demo of DAN, a bline super resolution method developed by Zhengxiong Luo et al. and presented in NeurIPS 2020. Try it by uploading an image."
examples = [["codes/config/DAN/examples/chip.png"]]
article = "Link to arxiv paper: [Unfolding the Alternating Optimization for " \
          "Blind Super Resolution](" \
          "https://arxiv.org/pdf/2010.02631.pdf)"

gr.Interface(predict, i, o, allow_flagging=False,
             analytics_enabled=False, title=title, description=description,
             examples=examples, article=article
             ).launch()

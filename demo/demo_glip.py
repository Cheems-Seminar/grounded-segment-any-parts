import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from maskrcnn_benchmark.config import cfg
from glip.predictor_glip import GLIPDemo


def show_predictions(scores, boxes, classes):
    num_obj = len(scores)
    if num_obj == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_obj))

    for obj_ind in range(num_obj):
        box = boxes[obj_ind]
        score = scores[obj_ind]
        name = classes[obj_ind]

        # color_mask = np.random.random((1, 3)).tolist()[0]
        color_mask = colors[obj_ind]

        # m = masks[obj_ind][0]
        # img = np.ones((m.shape[0], m.shape[1], 3))
        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        # ax.imshow(np.dstack((img, m*0.45)))

        x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color_mask, facecolor=(0, 0, 0, 0), lw=2))

        label = name + ': {:.2}'.format(score)
        ax.text(x0, y0, label, color=color_mask, fontsize='large', fontfamily='sans-serif')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segment-Anything-and-Name-It Demo", add_help=True)
    parser.add_argument(
        "--glip_checkpoint", type=str, default="glip_tiny.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--glip_config_file", type=str, default="glip/configs/glip_Swin_T_O365_GoldG.yaml", help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    # cfg
    glip_checkpoint = args.glip_checkpoint
    glip_config_file = args.glip_config_file

    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(glip_config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", glip_checkpoint])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    # initialize glip
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )

    # load image
    image = cv2.imread(image_path)

    # model inference
    scores, boxes, names = glip_demo.inference_on_image(image, text_prompt)

    # draw output image
    plt.figure(figsize=(10, 10))
    image_rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rbg)
    show_predictions(scores, boxes, names)

    plt.axis('off')
    image_name = image_path.split('/')[-1]
    plt.savefig(
        os.path.join(output_dir, "glip_output_{}".format(image_name)),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
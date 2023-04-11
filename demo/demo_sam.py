import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)



def show_anns(anns, ):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    num_show_obj = len([ann for ann in sorted_anns if ann['predicted_iou'] >= 1.0])
    num_show_obj = max(num_show_obj, 1)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_show_obj))

    polygons = []
    color = []
    show_obj_ind = 0
    for obj_ind, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        if ann['predicted_iou'] < 1.0:
            continue
        img = np.ones((m.shape[0], m.shape[1], 3))
        # color_mask = np.random.random((1, 3)).tolist()[0]
        if show_obj_ind == 0:
            color_mask = [1.0, 1.0, 1.0]
        else:
            color_mask = colors[show_obj_ind % num_show_obj]
        show_obj_ind += 1
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    # cfg
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # generate masks
    masks = mask_generator.generate(image)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)

    plt.axis('off')
    image_name = image_path.split('/')[-1]
    plt.savefig(
        os.path.join(output_dir, "sam_output_{}".format(image_name)),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

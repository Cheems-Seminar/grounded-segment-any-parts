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
from fastrcnn.modeling.clip_rcnn import build_clip_rcnn


def show_anns(anns, ):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        # print(ann['stability_score'], ann['predicted_iou'])
        if ann['predicted_iou'] < 1.0:
            continue
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        box = ann['bbox']
        x0, y0, w, h = box[0], box[1], box[2], box[3]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))


def show_anns_with_scores(img_size, anns, scores, text_prompt):
    if len(anns) == 0:
        return
    text_prompts = text_prompt.split(',')

    for ann, score in zip(anns, scores):
        ind = score.argmax()
        ann['clip_score'] = score[ind]
        ann['name'] = text_prompts[ind]
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img_h, img_w = img_size
    for ann in sorted_anns:
        box = ann['bbox']
        x0, y0, w, h = box[0], box[1], box[2], box[3]
        if w > 0.7 * img_w and h > 0.7 * img_h:
            continue
        clip_score = ann['clip_score']
        predicted_iou = ann['predicted_iou']
        # if clip_score < 0.5 or predicted_iou < 1.0:
        if clip_score < 0.5:
            continue
        # print(ann['clip_score'], ann['predicted_iou'])
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color_mask, facecolor=(0, 0, 0, 0), lw=2))
        label = ann['name'] + ': {:.2}'.format(clip_score)
        ax.text(x0, y0, label, color=color_mask, fontsize='large', fontfamily='sans-serif')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segment-Anything-and-Name-It Demo", add_help=True)
    parser.add_argument(
        "--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--clip_type", type=str, default="RN50x4", help="model type of clip"
    )

    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    # cfg
    sam_checkpoint = args.sam_checkpoint
    clip_type = args.clip_type
    image_path = args.input_image
    text_prompt = args.text_prompt
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

    # initialize CLIP
    clip_rcnn = build_clip_rcnn(clip_type)
    clip_rcnn.to(device=device)

    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # generate masks
    masks = mask_generator.generate(image)

    # prepare boxes
    boxes_xywh = [ann['bbox'] for ann in masks]
    boxes_xyxy = [[xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]] \
                  for xywh in boxes_xywh]
    # generate scores
    scores = clip_rcnn.forward_clip(image, boxes_xyxy, text_prompt)

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns_with_scores(image.shape[:2], masks, scores, text_prompt)

    plt.axis('off')
    image_name = image_path.split('/')[-1]
    plt.savefig(
        os.path.join(output_dir, "sani_output_{}".format(image_name)),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

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
import detectron2.data.transforms as T
import sys
sys.path.append('.')
from vlpart.vlpart import build_vlpart


def show_predictions(predictions, text_prompt):
    boxes = predictions.pred_boxes.tensor if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None

    if len(scores) == 0:
        return
    text_prompts = text_prompt.split('.')
    ax = plt.gca()
    ax.set_autoscale_on(False)
    num_show_obj = len([score for score in scores if score >= 0.7])
    num_show_obj = max(num_show_obj, 1)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_show_obj))

    show_obj_ind = 0
    for obj_ind in range(len(scores)):
        box = boxes[obj_ind]
        score = scores[obj_ind]
        name = text_prompts[classes[obj_ind]]
        if score < 0.7:
            continue

        # color_mask = np.random.random((1, 3)).tolist()[0]
        color_mask = colors[show_obj_ind % num_show_obj]
        show_obj_ind += 1

        x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color_mask, facecolor=(0, 0, 0, 0), lw=2))

        label = name + ': {:.2}'.format(score)
        ax.text(x0, y0, label, color=color_mask, fontsize='large', fontfamily='sans-serif')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--vlpart_checkpoint", type=str, default="swinbase_part_0a0000.pth", help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    # cfg
    vlpart_checkpoint = args.vlpart_checkpoint
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # initialize VLPart
    vlpart = build_vlpart(checkpoint=vlpart_checkpoint)
    vlpart.to(device=device)

    # load image
    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = T.ResizeShortestEdge([800, 800], 1333)
    height, width = original_image.shape[:2]
    image = preprocess.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    # model inference
    inputs = {"image": image, "height": height, "width": width}
    with torch.no_grad():
        predictions = vlpart.inference([inputs], text_prompt=text_prompt)[0]

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    if "instances" in predictions:
        instances = predictions["instances"].to('cpu')
        show_predictions(instances, text_prompt)

    plt.axis('off')
    image_name = image_path.split('/')[-1]
    plt.savefig(
        os.path.join(output_dir, "vlpart_output_{}".format(image_name)),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

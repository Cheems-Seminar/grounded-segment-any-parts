import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

import cv2
import matplotlib.pyplot as plt
import detectron2.data.transforms as T
from vlpart.vlpart import build_vlpart
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions


def show_predictions_with_masks(scores, boxes, classes, masks, text_prompt):
    num_obj = len(scores)
    if num_obj == 0:
        return
    text_prompts = text_prompt.split('.')
    ax = plt.gca()
    ax.set_autoscale_on(False)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_obj))

    for obj_ind in range(num_obj):
        box = boxes[obj_ind]
        score = scores[obj_ind]
        name = text_prompts[classes[obj_ind]]
        if score < 0.5:
            continue

        # color_mask = np.random.random((1, 3)).tolist()[0]
        color_mask = colors[obj_ind]

        m = masks[obj_ind][0]
        img = np.ones((m.shape[0], m.shape[1], 3))
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.45)))

        x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color_mask, facecolor=(0, 0, 0, 0), lw=2))

        label = name + ': {:.2}'.format(score)
        ax.text(x0, y0, label, color=color_mask, fontsize='large', fontfamily='sans-serif')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segment-Anything-and-Name-It Demo", add_help=True)
    parser.add_argument(
        "--vlpart_checkpoint", type=str, default="swinbase_part_0a0000.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="sam_vit_h_4b8939.pth", help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    # cfg
    vlpart_checkpoint = args.vlpart_checkpoint
    sam_checkpoint = args.sam_checkpoint
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

    # initialize SAM
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device=device))

    # load image
    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # vlpart model inference
    preprocess = T.ResizeShortestEdge([800, 800], 1333)
    height, width = original_image.shape[:2]
    image = preprocess.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    with torch.no_grad():
        predictions = vlpart.inference([inputs], text_prompt=text_prompt)[0]

    boxes, masks = None, None
    filter_scores, filter_boxes, filter_classes = [], [], []

    if "instances" in predictions:
        instances = predictions['instances'].to('cpu')
        boxes = instances.pred_boxes.tensor if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

        num_obj = len(scores)
        for obj_ind in range(num_obj):
            category_score = scores[obj_ind]
            if category_score < 0.7:
                continue
            filter_scores.append(category_score)
            filter_boxes.append(boxes[obj_ind])
            filter_classes.append(classes[obj_ind])

    if len(filter_boxes) > 0:
        # sam model inference
        sam_predictor.set_image(original_image)

        boxes_filter = torch.stack(filter_boxes)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filter, original_image.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        # remove small disconnected regions and holes
        fine_masks = []
        for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
            fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
        masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
        masks = torch.from_numpy(masks)

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)

    if len(filter_boxes) > 0:
        show_predictions_with_masks(filter_scores, filter_boxes, filter_classes,
                                    masks.to('cpu'), text_prompt)

    plt.axis('off')
    image_name = image_path.split('/')[-1]
    plt.savefig(
        os.path.join(output_dir, "vlpart_sam_output_{}".format(image_name)),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

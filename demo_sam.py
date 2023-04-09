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


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model



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


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

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
        print(ann['stability_score'], ann['predicted_iou'])
        if ann['predicted_iou'] < 1.0:
            continue
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
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

    # # load image
    # image_pil, _ = load_image(image_path)
    #
    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # predictor = SamPredictor()
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # predictor.set_image(image)

    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # generate masks
    masks = mask_generator.generate(image)

    # masks, _, _ = predictor.predict_torch(
    #     point_coords=None,
    #     point_labels=None,
    #     boxes=None,
    #     multimask_output=True,
    # )

    # import pdb
    # pdb.set_trace()
    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    # for box, label in zip(boxes_filt, pred_phrases):
    #     show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

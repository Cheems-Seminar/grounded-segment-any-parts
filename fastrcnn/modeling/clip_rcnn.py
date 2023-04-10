# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.layers import ShapeSpec, batched_nms, cat, nonzero_tuple, Conv2d, ConvTranspose2d
from detectron2.structures import ImageList, Instances, BitMasks, ROIMasks
import detectron2.utils.comm as comm

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, ROIMasks, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom

import clip
from .text_encoder import build_text_encoder


def build_clip_rcnn(clip_type='RN50'):
    pooler_res_dict = {
        "RN50": 14,
        "RN50x4": 18,
        "RN50x16": 24,
        "RN50x64": 28,
    }
    pooler_resolution = pooler_res_dict[clip_type]
    clip_rcnn = CLIP_RCNN(clip_type=clip_type, pooler_resolution=pooler_resolution)
    clip_rcnn.eval()
    return clip_rcnn


def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'


class CLIP_RCNN(nn.Module):
    def __init__(
        self,
        clip_type,
        pooler_resolution,
        pooler_scales: int = 16,
        sampling_ratio: int = 0,
        pooler_type: str = "ROIAlignV2",
        canonical_box_size: int = 224,
        softmax_t: float = 0.01,
    ):
        super().__init__()
        self.register_buffer("pixel_mean_clip",
            torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std_clip",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1), False)

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=(1.0 / pooler_scales, ),
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_box_size=canonical_box_size,
        )

        self.clip_model, self.clip_preprocess = clip.load(clip_type, device='cpu')
        self.softmax_t = softmax_t

        self.text_encoder = build_text_encoder(pretrain=True, visual_type=clip_type)

    @property
    def device(self):
        return self.pixel_mean_clip.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean_clip)

    def forward_clip(self, image, boxes, text_prompt):
        imageList_clip = self.preprocess_image_clip(image)
        features = self.clip_res_c4_backbone(imageList_clip.tensor)
        text_embed = self.get_text_embeddings(text_prompt)
        clip_scores = self.clip_res5_roi_heads(features, boxes, text_embed)
        return clip_scores.cpu()

    def get_text_embeddings(self, vocabulary, prefix_prompt='a '):
        vocabulary = vocabulary.split(',')
        texts = [prefix_prompt + x.lower().replace(':', ' ') for x in vocabulary]
        texts_aug = texts + ['background']
        emb = self.text_encoder(texts_aug).permute(1, 0)
        emb = F.normalize(emb, p=2, dim=0)
        return emb

    def clip_res_c4_backbone(self, tensor):
        def stem(x):
            x = resnet.relu1(resnet.bn1(resnet.conv1(x)))
            x = resnet.relu2(resnet.bn2(resnet.conv2(x)))
            x = resnet.relu3(resnet.bn3(resnet.conv3(x)))
            x = resnet.avgpool(x)
            return x
        resnet = self.clip_model.visual

        tensor = tensor.type(resnet.conv1.weight.dtype)
        tensor = stem(tensor)
        tensor = resnet.layer1(tensor)
        tensor = resnet.layer2(tensor)
        tensor = resnet.layer3(tensor)
        return tensor

    def clip_res5_roi_heads(self, features, boxes, text_embed):
        pro_boxes = self._move_to_current_device(Boxes(torch.tensor(boxes)))
        crop_features = self.pooler([features], [pro_boxes])
        box_features = self.clip_model.visual.layer4(crop_features)
        region_features = self.clip_model.visual.attnpool(box_features)
        region_features = F.normalize(region_features, p=2, dim=-1)

        similarity = ((1 / self.softmax_t) * region_features @ text_embed).softmax(dim=-1)
        clip_scores = similarity[:,:-1]
        return clip_scores


    def preprocess_image_clip(self, images):
        if not isinstance(images, list):
            images = [images]
        original_images = [self._move_to_current_device(torch.from_numpy(x).permute(2,0,1)) for x in images]
        images = [(x / 255.0 - self.pixel_mean_clip) / self.pixel_std_clip for x in original_images]
        images = ImageList.from_tensors(images,)
        return images


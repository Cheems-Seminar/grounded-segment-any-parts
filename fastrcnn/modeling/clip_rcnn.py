# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from skimage import color

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

from cc_torch import connected_components_labeling
from ..utils.adelaidet import aligned_bilinear, unfold_wo_center, mask_to_box
from .rcnn import GeneralizedRCNNX
from ..group_heads.rag import construct_finest_partition_watershed, rag_boundary, merge_hierarchy, hierarchy2segments
import higra as hg

from .hed_grouping import HED_Grouping
from .rcnn import GeneralizedRCNNX
import clip
from PIL import Image

@META_ARCH_REGISTRY.register()
class HED_Grouping_CLIP(HED_Grouping):
    @configurable
    def __init__(
        self,
        *,
        pooler: ROIPooler,
        clip_type: str = 'cuda',
        category_weight_path: str = 'datasets/metadata/coco_clip_a+cname.npy',
        image_scales: list = (0.5, 1.0, 2.0),
        crop_box_scales: list = (1.0, 1.5),
        softmax_t: float = 0.01,
        category_score_thresh: float = 0.01,
        category_iou_thresh: float = 0.5,
        category_box_topk: int = 100,
        class_filter: bool = False,
        one_box_per_class: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.register_buffer("pixel_mean_clip",
            torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std_clip",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1), False)
        self.pooler = pooler
        self.crop_box_scales = crop_box_scales
        self.image_scales = image_scales

        self.clip_model, self.clip_preprocess = clip.load(clip_type, device='cpu')
        self.softmax_t = softmax_t

        # (dim, #category)
        if category_weight_path.endswith('npy'):
            category_weight = np.load(category_weight_path)
            category_weight = torch.tensor(
                category_weight, dtype=torch.float32).permute(1, 0).contiguous()
        elif category_weight_path.endswith('pth'):
            category_weight = torch.load(category_weight_path,
                map_location="cpu").clone().detach().permute(1, 0).contiguous()
        else:
            raise NotImplementedError

        category_weight = F.normalize(category_weight, p=2, dim=0)
        self.register_buffer('category_weight', category_weight)

        self.category_score_thresh = category_score_thresh
        self.category_iou_thresh = category_iou_thresh
        self.category_box_topk = category_box_topk
        self.class_filter = class_filter
        self.one_box_per_class = one_box_per_class

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            "clip_type": cfg.MODEL.RECOGNITION_HEADS.CLIP_TYPE,
            "category_weight_path": cfg.MODEL.RECOGNITION_HEADS.CATEGORY_WEIGHT_PATH,
            "crop_box_scales": cfg.MODEL.RECOGNITION_HEADS.CROP_BOX_SCALES,
            "image_scales": cfg.MODEL.RECOGNITION_HEADS.IMAGE_SCALES,
            "softmax_t": cfg.MODEL.RECOGNITION_HEADS.SOFTMAX_T,
            "category_score_thresh": cfg.MODEL.RECOGNITION_HEADS.BOX_SCORE_THRESH,
            "category_iou_thresh": cfg.MODEL.RECOGNITION_HEADS.BOX_IOU_THRESH,
            "category_box_topk": cfg.MODEL.RECOGNITION_HEADS.BOX_TOPK,
            "class_filter": cfg.MODEL.CLASS_FILTER,
            "one_box_per_class": cfg.MODEL.ONE_BOX_PER_CLASS,
        })
        assert ret["image_scales"][0] <= ret["image_scales"][-1]  # ascending order
        pooler_resolution = cfg.MODEL.RECOGNITION_HEADS.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.RECOGNITION_HEADS.POOLER_TYPE
        pooler_scales     = tuple(1.0 / cfg.MODEL.RECOGNITION_HEADS.POOLER_SCALES / k
                             for k in ret["image_scales"])
        sampling_ratio    = cfg.MODEL.RECOGNITION_HEADS.POOLER_SAMPLING_RATIO
        canonical_box_size= cfg.MODEL.RECOGNITION_HEADS.CANONICAL_BOX_SIZE
        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_box_size=canonical_box_size,
        )
        return ret

    def forward(self, batched_inputs, do_postprocess=True):
        assert not self.training, 'HED_Grouping only supports inference mode'
        assert len(batched_inputs) == 1, "HED_Grouping only supports one image in one batch"

        if self.eval_grouping:
            return self.evaluation_clip(batched_inputs)

        imageList = self.preprocess_image(batched_inputs)
        hed_results = self.forward_hed(batched_inputs, imageList.tensor)
        if self.vis_period:
            return hed_results

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        levels_proposals = []
        for edge_level in self.edge_levels:
            proposals = self.edge_to_boxes(
                batched_inputs,
                1.0 - hed_results[edge_level],
                [x.image_size for x in gt_instances]
            )
            levels_proposals.append(proposals)
        grouping_proposals = self.merge_levels(levels_proposals, imageList.image_sizes)

        # merge to gt annotations
        pseudo_targets = []
        for image_idx in range(len(batched_inputs)):
            pseudo_target = Instances(gt_instances[image_idx].image_size)
            pseudo_boxes = grouping_proposals[image_idx].proposal_boxes.tensor
            pseudo_masks = grouping_proposals[image_idx].proposal_masks
            pseudo_scores = grouping_proposals[image_idx].scores

            # nms between gt_instances and pseudo_instances
            gt_per_image = gt_instances[image_idx]
            gt_per_image = gt_per_image[gt_per_image.gt_classes < self.num_classes]
            gt_boxes = gt_per_image.gt_boxes
            if len(gt_boxes) > 0:
                match_iou_matrix = pairwise_iou(gt_boxes, Boxes(pseudo_boxes))
                iou = torch.max(match_iou_matrix, dim=0)[0]
                keep = iou < self.iou_by_gt_thresh
                if keep.sum() < 1:
                    pseudo_target.pred_boxes = gt_per_image.gt_boxes
                    pseudo_target.pred_classes = torch.ones_like(gt_per_image.gt_classes)
                    pseudo_target.pred_global_masks = gt_per_image.gt_masks.tensor.float()
                    pseudo_targets.append(pseudo_target)
                    continue

                pseudo_boxes = pseudo_boxes[keep]
                pseudo_masks = pseudo_masks[keep]
                pseudo_scores = pseudo_scores[keep]

            # select topk pseudo_instances and merge to gt_instances
            _, idx = pseudo_scores.sort(descending=True)
            pseudo_boxes = pseudo_boxes[idx[:self.box_topk]]
            pseudo_masks = pseudo_masks[idx[:self.box_topk]]
            pseudo_scores = pseudo_scores[idx[:self.box_topk]]

            if len(gt_boxes) > 0:
                pseudo_target.pred_boxes = Boxes.cat([gt_per_image.gt_boxes, Boxes(pseudo_boxes)])
                pseudo_target.pred_classes = torch.cat(
                    [torch.ones_like(gt_per_image.gt_classes), torch.ones_like(pseudo_scores, dtype=torch.int64)],
                    dim=0)
                pseudo_target.pred_global_masks = torch.cat([gt_per_image.gt_masks.tensor, pseudo_masks]).float()
            else:
                pseudo_target.pred_boxes = Boxes(pseudo_boxes)
                pseudo_target.pred_classes = torch.ones_like(pseudo_scores, dtype=torch.int64)
                pseudo_target.pred_global_masks = pseudo_masks.float()

            pseudo_targets.append(pseudo_target)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            max_shape = imageList.tensor.shape[2:]
            return GeneralizedRCNNX._postprocess(pseudo_targets, batched_inputs, imageList.image_sizes, max_shape)
        else:
            return results


    def evaluation_clip(self, batched_inputs, do_postprocess=True):
        # get region proposals
        imageList = self.preprocess_image(batched_inputs)
        hed_results = self.forward_hed(batched_inputs, imageList.tensor)
        levels_proposals = []
        for edge_level in self.edge_levels:
            proposals = self.edge_to_boxes(batched_inputs,
                1.0 - hed_results[edge_level], imageList.image_sizes)
            levels_proposals.append(proposals)
        region_proposals = self.merge_levels(levels_proposals, imageList.image_sizes)

        # get CLIP scores
        imageList_clip = self.preprocess_image_clip(batched_inputs)
        image_tensor = imageList_clip.tensor
        height, width = image_tensor.shape[-2:]
        features_pyramid = []
        for image_scale in self.image_scales:
            scaled_tensor = F.interpolate(image_tensor, size=(int(height/image_scale), int(width/image_scale)),
                                          mode='bilinear', align_corners=False)
            features = self.clip_res_c4_backbone(scaled_tensor)
            features_pyramid.append(features)
        clip_scores = self.clip_res5_roi_heads(features_pyramid, region_proposals)

        results = self.clip_inference(batched_inputs, region_proposals, clip_scores)
        if self.vis_period:
            self.visualize_clip(batched_inputs, results)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNNX._postprocess(results, batched_inputs,
                                                 imageList.image_sizes, imageList.tensor.shape[2:])
        else:
            return results

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

    def clip_res5_roi_heads(self, features, proposals):
        proposal_boxes = [x.proposal_boxes for x in proposals]
        if sum([len(x) for x in proposal_boxes]) < 1:
            dummy_scores = torch.ones(
                (0, self.category_weight.shape[-1]),
                device=self.device) / self.category_weight.shape[-1]
            clip_scores = [dummy_scores for _ in proposal_boxes]
            return clip_scores

        region_features = []
        for boxScale in self.crop_box_scales:
            scaled_proposal_boxes = self.scale_proposal_boxes(proposal_boxes, boxScale)
            crop_features = self.pooler(features, scaled_proposal_boxes)
            box_features = self.clip_model.visual.layer4(crop_features)
            cur_feats = self.clip_model.visual.attnpool(box_features)
            cur_feats /= cur_feats.norm(dim=-1, keepdim=True)
            region_features.append(cur_feats)

        region_features = torch.stack(region_features, dim=0).sum(0) / len(self.crop_box_scales)
        region_features /= region_features.norm(dim=-1, keepdim=True)

        similarity = ((1 / self.softmax_t) * region_features @ self.category_weight).softmax(dim=-1)

        num_inst_per_image = [len(p) for p in proposals]
        clip_scores = similarity.split(num_inst_per_image, dim=0)

        return clip_scores

    def clip_inference(self, batched_inputs, proposal_boxes, clip_scores):
        pseudo_targets = []
        for image_idx in range(len(batched_inputs)):
            category_scores = clip_scores[image_idx]
            if len(category_scores) < 1:
                pseudo_targets.append(self.dummy_instance(proposal_boxes[image_idx]))
                continue
            pro_boxes = proposal_boxes[image_idx].proposal_boxes
            pro_masks = proposal_boxes[image_idx].proposal_masks.float()
            pro_cat_scores, pro_classes = torch.max(category_scores, dim=1)
            if self.class_filter:
                ann_classes = batched_inputs[image_idx]["instances"].gt_classes.to(self.device).unique()
                if len(ann_classes) < 1:
                    pseudo_targets.append(self.dummy_instance(proposal_boxes[image_idx][:0]))
                    continue
                if self.one_box_per_class:
                    pro_cat_scores, pro_ind = torch.max(category_scores[:, ann_classes], dim=0)
                    pro_boxes = pro_boxes[pro_ind]
                    pro_masks = pro_masks[pro_ind]
                    pro_classes = ann_classes
                else:
                    filter_mask = (pro_classes[:, None] - ann_classes[None, :] == 0).sum(-1) > 0
                    pro_boxes = pro_boxes[filter_mask]
                    pro_masks = pro_masks[filter_mask]
                    pro_classes = pro_classes[filter_mask]
                    pro_cat_scores = pro_cat_scores[filter_mask]

            filter_mask = pro_cat_scores > self.category_score_thresh
            pseudo_target = Instances(proposal_boxes[image_idx].image_size)
            pseudo_target.pred_boxes = pro_boxes[filter_mask]
            pseudo_target.pred_global_masks = pro_masks[filter_mask]
            pseudo_target.pred_classes = pro_classes[filter_mask]
            pseudo_target.scores = pro_cat_scores[filter_mask]

            pseudo_targets.append(pseudo_target)
        return pseudo_targets

    def visualize_clip(self, batched_inputs, pseudo_targets):
        import os
        import cv2
        import numpy as np
        import time
        from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
        from colormap import colormap
        color_maps = colormap()

        os.makedirs('output_clip', exist_ok=True)
        for image_idx in range(len(batched_inputs)):
            image_show = batched_inputs[image_idx]["image"].permute(1, 2, 0).numpy()
            height, width = image_show.shape[:2]
            image_show = image_show.copy()

            pseudo_target = pseudo_targets[image_idx]
            pseudo_boxes = pseudo_target.pred_boxes.tensor
            pseudo_mmasks = pseudo_target.pred_global_masks
            pseudo_classes = pseudo_target.pred_classes
            pseudo_scores = pseudo_target.scores

            for instance_id, (bbox, mmask, sscore, llable) in \
                    enumerate(zip(pseudo_boxes, pseudo_mmasks, pseudo_scores, pseudo_classes)):
                instance_color = color_maps[instance_id % len(color_maps)]
                image_show = cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                           instance_color.tolist(), 2)
                mask_ = mmask.cpu().numpy().astype(np.bool_)[:height, :width]
                for color_channel in range(3):
                    image_show[mask_, color_channel] = \
                        image_show[mask_, color_channel] * 0.5 + instance_color[color_channel] * 0.5
                label_text = COCO_CATEGORIES[llable]["name"]
                cv2.putText(image_show, "{} {:.2f}".format(label_text, sscore),
                            (int(bbox[0] - 5), int(bbox[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imwrite("output_clip/{}.png".format(
                batched_inputs[image_idx]['file_name'].split('/')[-1][:-4]), image_show)

    def scale_proposal_boxes(self, proposal_boxes, scale=1.0):
        scaled_proposal_boxes = []
        for boxes_per_image in proposal_boxes:
            boxes_tensor = boxes_per_image.tensor
            x0, y0 = boxes_tensor[:, 0], boxes_tensor[:, 1]
            x1, y1 = boxes_tensor[:, 2], boxes_tensor[:, 3]
            cx, cy = ((x0 + x1) / 2), ((y0 + y1) / 2)
            bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
            new_x0, new_y0 = cx - bw / 2, cy - bh / 2
            new_x1, new_y1 = cx + bw / 2, cy + bh / 2
            scaled_boxes_tensor = torch.stack([new_x0, new_y0, new_x1, new_y1], dim=1)
            scaled_proposal_boxes.append(Boxes(scaled_boxes_tensor))
        return scaled_proposal_boxes

    def dummy_instance(self, target):
        dummy_target = Instances(target.image_size)
        dummy_target.pred_boxes = target.proposal_boxes
        dummy_target.pred_global_masks = target.proposal_masks.float()
        dummy_target.scores = target.scores
        dummy_target.pred_classes = torch.zeros_like(target.scores, dtype=torch.int64)
        return dummy_target

    def rcnn_clip_score(self, batched_inputs, proposal_boxes):
        clip_scores = []
        for img_id in range(len(batched_inputs)):
            input_img = batched_inputs[img_id]['image'].to(torch.uint8).permute(1, 2, 0)
            if self.input_format == 'BGR':
                input_img = input_img[:, :, [2, 1, 0]]
            height, width = input_img.shape[:2]
            pilImg = Image.fromarray(input_img.numpy())  # RGB

            pro_boxes = proposal_boxes[img_id].pred_boxes.tensor.cpu().numpy()
            if len(pro_boxes) < 1:
                clip_scores.append(torch.ones(
                    (0, self.category_weight.shape[-1]),
                    device=self.device) / self.category_weight.shape[-1])
                continue

            pro_features = []
            for boxScale in self.crop_box_scales:
                curInputList = []
                for b_idx, box in enumerate(pro_boxes):
                    scaledBox = scale_box(box, boxScale, max_H=height, max_W=width)
                    cropImg = pilImg.crop(scaledBox)
                    clipInput = self.clip_preprocess(cropImg).unsqueeze(0).to(self.device)
                    curInputList.append(clipInput)
                curInputBatch = torch.cat(curInputList, dim=0)
                with torch.no_grad():
                    curImgFeat = self.clip_model.encode_image(curInputBatch)
                    curImgFeat /= curImgFeat.norm(dim=-1, keepdim=True)
                pro_features.append(curImgFeat)

            image_features = torch.stack(pro_features, dim=0).sum(0) / len(self.crop_box_scales)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = ((1 / self.softmax_t) * image_features @ self.category_weight).softmax(dim=-1)
            clip_scores.append(similarity)

        return clip_scores

    def preprocess_image_clip(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        if self.input_format == 'BGR':
            original_images = [ori_img[[2, 1, 0], :, :] for ori_img in original_images]
        images = [(x / 255.0 - self.pixel_mean_clip) / self.pixel_std_clip for x in original_images]
        images = ImageList.from_tensors(
            images,
            self.size_divisibility,
        )
        return images

#### for CLIP text embeddings
def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res

single_template = [
    'a photo of {article} {}.'
]
multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',

    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]


def build_text_embedding(model, categories, templates, add_this_is=False, show_process=True):
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        if show_process:
            print('Building text embeddings...')
        for catName in (tqdm(categories) if show_process else categories):
            texts = [template.format(catName, article=article(catName)) for template in templates]
            if add_this_is:
                texts = ['This is ' + text if text.startswith('a') or text.startswith('the') else text
                         for text in texts]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()

            text_embeddings = model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()

        return all_text_embeddings.transpose(dim0=0, dim1=1)

def scale_box(box, scale, max_H=np.inf, max_W=np.inf):
    # box: x0, y0, x1, y1
    # scale: float
    x0, y0, x1, y1 = box

    cx = ((x0 + x1) / 2)
    cy = ((y0 + y1) / 2)
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale

    new_x0 = max(cx - bw / 2, 0)
    new_y0 = max(cy - bh / 2, 0)
    new_x1 = min(cx + bw / 2, max_W)
    new_y1 = min(cy + bh / 2, max_H)

    return [new_x0, new_y0, new_x1, new_y1]
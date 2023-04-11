## CLIP Fast RCNN
CLIP Fast RCNN uses the output of Segment-Anything model as region proposals, and use a shared feature map from CLIP to perform RoI-Pool.

## Run Demo
```
# CLIP model weight is downloaded by default
cd {root_repo}
python demo_sam_clip.py --input_image assets/demo.png --output_dir outputs_demo --text_prompt "dog"
```


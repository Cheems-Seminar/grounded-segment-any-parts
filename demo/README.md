## More Demos
We provide more demo choices.

## SAM
segment anything
```
cd {root_repo}
python demo/demo_sam.py --input_image assets/demo1.png --output_dir outputs_demo
```

## SAM + CLIP Fast R-CNN
segment anything, CLIP Fast R-CNN
```
cd {root_repo}
python demo/demo_sam_clip.py --input_image assets/demo1.png --output_dir outputs_demo --text_prompt "dog"
```

## GLIP
object-level detection
```
cd {root_repo}
python demo/demo_glip.py --input_image assets/demo1.png --output_dir outputs_demo --text_prompt "dog"
```

## VLPart
part-level detection
```
cd {root_repo}
python demo/demo_vlpart.py --input_image assets/demo1.png --output_dir outputs_demo --text_prompt "dog head. dog leg"
```
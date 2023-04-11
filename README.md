

# Segment Anything and Name It


![](./assets/teaser.png)



## :sparkles::sparkles: Highlights :sparkles::sparkles:

- Segment anything
- Text Prompt at both object level and ***part*** level.
- Use ChatGPT as Controller


## Model Summary


| Model | Function |
| ----  | -------- |
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | Segment anything from prompt |
| [GLIP](https://github.com/microsoft/GLIP) | Grounded language-image pre-training |
| [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) | Connects ChatGPT and segmentation foundation models |
| :star:**VLPart**:star: (*under review*) | Going denser with open-vocabulary part segmentation |

## Usage

### Install

See [installation instructions](INSTALL.md).



### :robot: Play with ChatBot

```bash
# prepare your private OpenAI key (for Linux)
export OPENAI_API_KEY={Your_Private_Openai_Key}
python chatbot.py --load "ImageCaptioning_cuda:0, SegmentAnything_cuda:1, PartPromptSegmentAnything_cuda:1, ObjectPromptSegmentAnything_cuda:0"
```

<img src="./assets/demo_chat_short.gif" width="600">


### :last_quarter_moon: Prompt Segment Anything at Part Level

```bash

wget https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

python demo_vlpart_sam.py --input_image assets/twodogs.jpeg --output_dir outputs_demo --text_prompt "dog head"
```
Result:

<img src="./assets/vlpart_sam_output_twodogs.jpeg" width="600">

### :full_moon: Prompt Segment Anything at Object Level

```bash
wget https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/glip_large.pth

python demo_glip_sam.py --input_image assets/demo2.jpeg --output_dir outputs_demo --text_prompt "frog"

```

Result:

<img src="./assets/glip_sam_output_demo2.jpeg" width="600">


###  :lollipop: Multi-Prompt

For multiple prompts, seperate each prompt with `.`, for example, `--text_prompt "dog head. dog nose"`


### Model Checkpoints

* [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* [VLPart Swin-Base](https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/swinbase_part_0a0000.pth)
* [GLIP Swin-Large](https://github.com/Cheems-Seminar/segment-anything-and-name-it/releases/download/v1.0/glip_large.pth)


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


## Acknowledgement

A large part of the code is borrowed from [segment-anything](https://github.com/facebookresearch/segment-anything), [CLIP](https://github.com/openai/CLIP), [GLIP](https://github.com/microsoft/GLIP), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt). Many thanks for their wonderful works.

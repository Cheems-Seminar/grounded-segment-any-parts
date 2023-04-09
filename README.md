# Segment Anything and Name It

## Installation

See [installation instructions](INSTALL.md).

## Run Segment-Anything Demo
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

python demo_sam.py --input_image assets/demo.png --output_dir outputs_demo 
```

## Run Segment-Anything-and-Name-It Demo
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# CLIP model weight is downloaded by default

python demo_sani.py --input_image assets/demo.png --output_dir outputs_demo --text_promt "dog"
```

## Run ChatBot Demo
```
python chatbot.py
```


## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of this project is licensed under a [MIT License](LICENSE). Portions of the project are available under separate license of the referred projects.


## Acknowledgement

A large part of the code is borrowed from [segment-anything](https://github.com/facebookresearch/segment-anything), [clip](https://github.com/openai/CLIP), [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), [Visual-ChatGPT](https://github.com/microsoft/visual-chatgpt). Many thanks for their wonderful works.

## Citation

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```BibTeX

@misc{cheems2023sani,
  author =       {Cheems},
  title =        {Segment Anything and Name It},
  howpublished = {\url{https://github.com/cheems-lab/segment-anything-and-name-it}},
  year =         {2023}
}
```

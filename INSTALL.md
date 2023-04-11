
```bash

git clone https://github.com/Cheems-Seminar/segment-anything-and-name-it.git
cd segment-anything-and-name-it
conda create -n sani python=3.8
conda activate sani

# 1. Install torch 1.12.1
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge


# 2. Install requirements
pip install -r requirements.txt


# 3. Install sam
cd segment-anything; pip install -e .; cd ..

# 4. Install glip
cd glip; pip install -e .; cd ..
```

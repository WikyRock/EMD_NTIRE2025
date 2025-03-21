# Dual Channel Cross-modal Mamba for Event-based Motion Deblurring
`for NTIRE 2025 Event-Based Image Deblurring Challenge Factsheet`


[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Thank you for your arriving here!!!

### 1 Before using the code, you are advised to install `Mamba` related drivers and virtual environment.

```bash
conda create -n your_env_name python=3.10.13
conda activate your_env_name
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
```

### 2 The `Ver1.1.1` Conv-1d and Mamba-ssm are preferred.  You could download their zips and install them manually.
Here are the Link:

[Github causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases)

[Github mamba-ssm](https://github.com/state-spaces/mamba/releases)

### 3 You could download the zip or clone our programe :
```bash
git clone https://github.com/WikyRock/EMD_NTIRE2025.git
cd EMD_NTIRE2025-master
```

### 4 In order to use the mamba module correctly, please use the `selective_scan_interface.py` in document `modules` to cover the original version in the installed mamba dependencies.


### 5 download pretrained model pth document to `models` and load`demo_test.py` with changing the path of HighREV tset dataset. 

Here is the link: [Google Drive](https://drive.google.com/drive/folders/1x9f8-q7mFggnCsx0TyOkK6T79K3mOt5g?usp=drive_link) 
or [Baidu Drive](https://pan.baidu.com/s/1P_1UKENeKXxSJKysQlTcCw) Password: 1314

and you can download our original test result: [Google Drive](https://drive.google.com/drive/folders/1x9f8-q7mFggnCsx0TyOkK6T79K3mOt5g?usp=drive_link)  or  [Baidu Drive](https://pan.baidu.com/s/1n4EDpjyeO6h9Pf1rpMIgjg) Password: 1314


### Acknowledgement

Thanks to the inspirations and codes from [REDNet](https://github.com/xufangchn/Motion-Deblurring-with-Real-Events.git) and [Pan-Mamba](git@github.com:alexhe101/Pan-Mamba.git)


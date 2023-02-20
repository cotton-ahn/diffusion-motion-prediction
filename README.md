# Can We Use Diffusion Probabilistic Models for 3D Motion Prediction?
A repository of a paper named "Can We Use Diffusion Probabilistic Models for 3D Motion Prediction?", accepted to ICRA 2023.

## Installation
- Tested Environment
    * Ubuntu 20.04
    * Python 3.9 with Anaconda
1. Clone this repository, and move to root of this repo.
2. Prepare [human3.6m](https://www.cs.stanford.edu/people/ashesh/h3.6m.zip) dataset for euler angle based experiments. Save Folder `h3.6m/` into `./data`.
3. Activate your conda environment, train pytorch based on your system, run `pip install -r requirements.txt`.
    * In my case, since I use cuda=11.6, I used command `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia` to install pytorch.
4. Clone [DLow]('https://github.com/Khrylx/DLow') repository.
5. Prepare dataset (human3.6m and humaneva) as [DLow]('https://github.com/Khrylx/DLow') suggests, and put `*.npz` files to `./data` and `./DLow/data`.
6. Copy `*.yml` files in `cfg/dlow` of our repository into `DLow/motion_pred/cfg`, by `cp cfg/dlow/*.yml DLow/motion_pred/cfg`.

## Preprocess
- Before conducting experiments with Human 3.6M dataset, you need to run `python preprocess.py` first.
- After running the code, `h36m_euler.pkl` will be created in `./data`.

### Training


### Evaluation


### Make comparison with DLow



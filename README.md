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
6. Copy `./src/dlow/eval.py` of our repository into `DLow/motion_pred/eval.py`.

## Preprocess
- Before conducting experiments with Human 3.6M dataset, you need to run `python preprocess.py` first.
- After running the code, `h36m_euler.pkl` will be created in `./data`.

### Training


### Evaluation


### Make comparison with DLow
- DLow codes are so well written! So (1) adding our config files and evalataion code to cloned DLow repository and (2) running the DLow code in DLow repository is enough. Below is the detailed process.

1. 


### todo list
- finish training the code and get results
- get more visualization results for adding it to webpage 
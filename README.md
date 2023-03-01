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

## Training
```
python train.py --cfg h36m_euler_series_20step
python train.py --cfg h36m_euler_parallel_20step
python train.py --cfg h36m_xyz_series_20step
python train.py --cfg h36m_xyz_parallel_20step
python train.py --cfg humaneva_xyz_series_20step
python train.py --cfg humaneva_xyz_parallel_20step
```
* Your log will be stored to `./log/[TRAIN]*.txt`.

## Quantitative Evaluation
* For deterministic prediction, run below code
    ```
    python eval.py --cfg h36m_euler_series_20step --mode stats
    python eval.py --cfg h36m_euler_parallel_20step --mode stats
    python eval.py --cfg h36m_xyz_series_20step --mode stats
    python eval.py --cfg h36m_xyz_parallel_20step --mode stats
    python eval.py --cfg humaneva_xyz_series_20step --mode stats
    python eval.py --cfg humaneva_xyz_parallel_20step --mode stats
    ```

## Qualitative Evaluation
* For plotting the prediction video, run below codes.
    ```
    python eval.py --cfg h36m_euler_series_20step --mode viz
    python eval.py --cfg h36m_euler_parallel_20step --mode viz
    python eval.py --cfg h36m_xyz_series_20step --mode viz
    python eval.py --cfg h36m_xyz_parallel_20step --mode viz
    python eval.py --cfg humaneva_xyz_series_20step --mode viz
    python eval.py --cfg humaneva_xyz_parallel_20step --mode viz
    ```

* For plotting the figure looks similar as we present in paper, please run below code.
    ```
    python first_figure.py --cfg h36m_euler_series_20step
    python first_figure.py --cfg h36m_euler_parallel_20step
    ```

* For visualizing how the noise becomes the data from the reverse process, run below code.
    ```
    python viz_diff_process.py --cfg h36m_euler_series_20step
    python viz_diff_process.py --cfg h36m_euler_parallel_20step
    ```

## Pretrained model
* Please find our pre-trained model on `trained_models/dataName_poseType_modelType_stepSize/`

## Make comparison with DLow
- DLow codes are so well written! So (1) adding our config files and evalataion code to cloned DLow repository and (2) running the DLow code in DLow repository is enough. Below is the detailed process.

1. Copy `./src/dlow/eval.py` of our repository into `DLow/motion_pred/eval.py`.
2. Copy your preprocessed data (i.e., data_3d_humaneva15.npz) into `DLow/data`.
3. Run DLow's training code as their README file instructs.
    ```
    python motion_pred/exp_vae.py --cfg h36m_nsamp50
    python motion_pred/exp_vae.py --cfg humaneva_nsamp50
    python motion_pred/exp_dlow.py --cfg h36m_nsamp50
    python motion_pred/exp_dlow.py --cfg humaneva_nsamp50
    ```
4. Run DLow's testing code as their README file instructs.
    ```
    python motion_pred/eval.py --cfg h36m_nsamp50 model stats
    python motion_pred/eval.py --cfg humaneva_nsamp50 model stats
    ```

5. To compare visualization between DLow and Diffusion model, run below code at the root of our repo.
    ```
    python viz_comp_dlow.py
    ```
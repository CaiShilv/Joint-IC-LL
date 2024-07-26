# Make Lossy Compression Meaningful for Low-Light Images

This is a pytorch project for the paper **Make Lossy Compression Meaningful for Low-Light Images** by Shilyu Cai, Liqun Chen, Sheng Zhong, Luxin Yan, Jiahuan Zhou, and Xu Zou presented at **AAAI 2024**.


## Introduction
We propose a novel joint solution to simultaneously achieve a high compression rate and good enhancement performance for low-light images with much lower computational cost and fewer model parameters.

[paper link](https://ojs.aaai.org/index.php/AAAI/article/view/28664)

## dataset

### SID and SMID dataset
For SID, we use the subset captured by the Sony camera and follow the script provided by SID to transfer the low-light images from RAW to RGB using rawpy’s default ISP. 

For SMID, we use its full images and also transfer the RAWdata to RGB, since our work explores low-light image enhancement in the RGB domain.

You can download the processed datasets for SID and SMID from [baidu pan](https://pan.baidu.com/s/1HRr-5LJO0V0CWqtoctQp9w) (验证码: gplv) and [baidu pan](https://pan.baidu.com/s/1Qol_4GsIjGDR8UT9IRZbBQ) (验证码: btux), and there should contain "SMID_Long_np and SMID_LQ_np" and "long_sid2 and short_sid2".

### SDSD dataset
Different from original SDSD datasets with dynamic scenes, we utilize its static version (the scenes are the same of original SDSD).

And you can download the SDSD-indoor and SDSD-outdoor from [baidu pan](https://pan.baidu.com/s/1rfRzshGNcL0MX5soRNuwTA) (验证码: jo1v) and [baidu pan](https://pan.baidu.com/s/1JzDQnFov-u6aBPPgjSzSxQ) (验证码: uibk), and there should contain "indoor_static_np" and "outdoor_static_np".

**Note:** It is more recommended that you download the dataset ([baidu pan]() (验证码: )) converted and integrated with my comments.

## Project Setup

First install Python 3. We advise you to install Python 3 and PyTorch with Anaconda:

```
conda create --name Joint_IC_LL python=3.8
conda activate Joint_IC_LL
```

Clone the repo and install the complementary requirements:
```
pip install -r requirements.txt
```

## Usage

### Train
Since compression tasks typically involve one model adapting to all datasets, we integrate training data from multiple datasets together for training using the train config.
For example:

```
OMP_NUM_THREADS=4 python -m torch.distributed.launch --use_env --master_port=6167 --nproc_per_node=4 train.py -opt=./options/train/all_data_combine_psnr.yml --launcher=pytorch
```

### Test

We use PSNR and SSIM as the metrics for evaluation. Evaluate the model on the corresponding dataset using the test config.

For the evaluation on indoor subset of SDSD-indoor, you should write the location of checkpoint in ''pretrain_model_G'' of options/test/all_data_combine_psnr.yml. It is worth noting that you should modify the data path(**"dataroot_GT"** and **"dataroot_LQ"**) and image resolution(**"train_size"**) in the configuration file.
use the following command line:

```
python test.py -opt=./options/test/all_data_combine_psnr.yml -dataset_name=SDSD-indoor
```

### Pre-trained Model

You can download our trained model using the following links: [baidu pan](https://pan.baidu.com/s/19icSarveBqvTlphqEy6TDQ)(验证码: tjps).

## Citation Information

If you find the project useful, please cite:

```
@inproceedings{cai2024JointICLL,
  title={Make Lossy Compression Meaningful for Low-Light Images},
  author={Shilyu Cai, Liqun Chen, Sheng Zhong, Luxin Yan, Jiahuan Zhou and Xu Zou},
  booktitle={AAAI},
  year={2024}
}
```


## Acknowledgments
This source code is inspired by [SNR-Aware-Low-Light-Enhance](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).

## Contributions
If you have any questions/comments/bug reports, feel free to e-mail the author Shilyu Cai ([caishilv@hust.edu.cn](caishilv@hust.edu.cn) or [caishilv1024@gmail.com](caishilv1024@gmail.com)).

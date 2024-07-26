import glob
import os
import ipdb
import cv2
import torch
import math
import tqdm
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

def glob_file_np_list(root):
    return sorted(glob.glob(os.path.join(root, '*.npy')))


def glob_file_png_list(root):
    return sorted(glob.glob(os.path.join(root, '*.png')))


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    # print(torch.max(tensor))
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.

    return img_np.astype(out_type)


def read_np_img(path, size=None):
    """read image by numpy.load()
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = np.load(path)
    if img is None:
        print(path)
    if size is not None:
        img = cv2.resize(img, (size[0], size[1]))
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_rgb_img(path, size=None):
    img = cv2.imread(path)
    return img


def read_muti_folder_img_seq(path, size=None):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """

    img_l = read_np_img(path, size)
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    try:
        imgs = imgs[:, :, [2, 1, 0]]
        # imgs = imgs[:, :, [2, 0, 1]] ×
        # imgs = imgs[:, :, [0, 2, 1]] ×
        # imgs = imgs[:, :, [0, 1, 2]] ×
        # imgs = imgs[:, :, [1, 2, 0]]
        # imgs = imgs[:, :, [1, 0, 2]] ×


    except Exception:
        import ipdb; ipdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (2, 0, 1)))).float()

    return imgs


def numpy_to_rgb(file_dir_pair: dict):
    for k_v in file_dir_pair:
        print("Begin {} .........".format(k_v))
        input_dir = file_dir_pair[k_v][0]
        output_dir = file_dir_pair[k_v][1]
        src_folder = glob_file_list(input_dir)
        src_folder_tqdm = tqdm.tqdm(src_folder)
        for c_src_folder in src_folder_tqdm:
            folder_base_name = os.path.basename(c_src_folder)
            src_file_list = glob_file_list(c_src_folder)
            for cur_file in src_file_list:
                images = read_muti_folder_img_seq(cur_file)
                c_basename = os.path.basename(cur_file)
                c_basename_pre, c_basename_fix = os.path.splitext(c_basename)
                output_folder = os.path.join(output_dir, folder_base_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                s_images = tensor2img(images, out_type=np.uint8, min_max=(0, 1))
                cv2.imwrite(os.path.join(output_folder, '{}.png'.format(c_basename_pre)), s_images)


def rgb_to_numpy(file_dir_pair: dict):
    for k_v in file_dir_pair:
        print("Begin {} .........".format(k_v))
        input_dir = file_dir_pair[k_v][1]
        output_dir = file_dir_pair[k_v][0]
        src_folder = glob_file_list(input_dir)
        src_folder_tqdm = tqdm.tqdm(src_folder)
        for c_src_folder in src_folder_tqdm:
            folder_base_name = os.path.basename(c_src_folder)
            src_file_list = glob_file_list(c_src_folder)
            for cur_file in src_file_list:
                images = read_rgb_img(cur_file)
                c_basename = os.path.basename(cur_file)
                c_basename_pre, c_basename_fix = os.path.splitext(c_basename)
                output_folder = os.path.join(output_dir, folder_base_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                np.save(os.path.join(output_folder, '{}.npy'.format(c_basename_pre)), images)


def main1():
    file_dir_pair_0 = {'SMID_LQ': ['xx/smid/full_smid/smid/SMID_LQ_np',
                            'xx/smid_RGB/SMID_LQ_np'],

                'SMID_Long': ['xx/smid/full_smid/smid/SMID_Long_np',
                              'xx/smid_RGB/SMID_Long_np'],

                'SDSD_indoor_GT': ['xx/full_sdsd_indoor_static/indoor_static_np/GT',
                                   'xx/sdsd_indoor_static_RGB/GT'],
                'SDSD_indoor_input': ['xx/full_sdsd_indoor_static/indoor_static_np/input',
                                      'xx/sdsd_indoor_static_RGB/input'],
                'SDSD_outdoor_GT': [
                    'xx/full_outdoor_static/outdoor_static_np/GT',
                    'xx/sdsd_outdoor_static_RGB/GT'],
                'SDSD_outdoor_input': ['xx/full_outdoor_static/outdoor_static_np/input',
                                       'xx/sdsd_outdoor_static_RGB/input'],
                'SID_long': ['xx/sid/sid_processed/long_sid2',
                             'xx/sid_RGB/long_sid2'],
                'SID_short': ['xx/sid/sid_processed/short_sid2',
                              'xx/sid_RGB/short_sid2']
                }
    file_dir_pair = {'SMID_LQ': ['xx/Test_RGB_to_numpy/smid/SMID_LQ_np',
                            'xx/smid_RGB/SMID_LQ_np'],

                'SMID_Long': ['xx/smid/SMID_Long_np',
                              'xx/smid_RGB/SMID_Long_np'],

                'SDSD_indoor_GT': ['xx/indoor_static_np/GT',
                                   'xx/sdsd_indoor_static_RGB/GT'],

                'SDSD_indoor_input': ['xx/indoor_static_np/input',
                                      'xx/sdsd_indoor_static_RGB/input'],

                'SDSD_outdoor_GT': ['xx/outdoor_static_np/GT',
                                    'xx/sdsd_outdoor_static_RGB/GT'],

                'SDSD_outdoor_input': ['xx/outdoor_static_np/input',
                                       'xx/sdsd_outdoor_static_RGB/input'],

                'SID_long': ['xx/sid_processed/long_sid2',
                             'xx/sid_RGB/long_sid2'],

                'SID_short': ['xx/sid_processed/short_sid2',
                              'xx/sid_RGB/short_sid2']
                }
    file_dir_pair_2 = {'test': ["xx/Two_pipeline/After_compress_results/test",
                                "xx/Two_pipeline/After_compress_results/RGB"]}
    # numpy_to_rgb(file_dir_pair_0)
    # rgb_to_numpy(file_dir_pair_2)

    single_sid_train = "xx/Two_pipeline/sdsd_outdoor_static/GT_train"
    combine_sid_train = "xx/combine_dataset/train/GT"

    single_sid_train_list = glob_file_list(single_sid_train)

    src_folder_tqdm = tqdm.tqdm(single_sid_train_list)
    for c_src_folder in src_folder_tqdm:                     
        folder_base_name = os.path.basename(c_src_folder)   
        src_file_list = glob_file_list(c_src_folder)       
        for cur_file in src_file_list:
            img = np.load(cur_file)
            c_basename = os.path.basename(cur_file)
            c_basename_pre, c_basename_fix = os.path.splitext(c_basename)
            t_folder_base_name = f'SDSD_outdoor_{folder_base_name}'
            output_folder = os.path.join(combine_sid_train, t_folder_base_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            np.save(os.path.join(output_folder, '{}.npy'.format(c_basename_pre)), img)


def main2():
    single_semi_train = "xx/CVPR-2020-Semi-Low-Light-Dataset/RGB/stage2-LOL/Our_low_test"
    combine_sid_train = "xx/CVPR-2020-Semi-Low-Light-Dataset/npy/stage2-LOL/Our_low_test"
    single_sid_train_list = glob_file_png_list(single_semi_train)   
    src_folder_tqdm = tqdm.tqdm(single_sid_train_list)
    index = 0
    for c_src_file in src_folder_tqdm:
        c_save_folder = f'SEMI_stage2_LOL_{index:05d}'
        images = read_rgb_img(c_src_file)
        c_basename = os.path.basename(c_src_file)
        c_basename_pre, c_basename_fix = os.path.splitext(c_basename)
        output_folder = os.path.join(combine_sid_train, c_save_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        np.save(os.path.join(output_folder, '{}.npy'.format(c_basename_pre)), images)
        index += 1
if __name__ == '__main__':
    # main1()
    main2()

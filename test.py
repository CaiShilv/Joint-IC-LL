import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
from PIL import Image

import utils.util as util
import data.util as data_util
from models import create_model

import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from models import pad
import time
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('-dataset_name', type=str, default='SID_test')
args=parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_imgs = True
    model = create_model(opt)
    model.netG.module.update(force=True)
    model_name = os.path.basename(opt['path']['pretrain_model_G'])
    save_folder = './experiments/test/{}/{}/{}_{}'.format(opt['name'], model_name, args.dataset_name, current_datetime)
    GT_folder = osp.join(save_folder, 'images/GT')
    output_folder = osp.join(save_folder, 'images/output')
    input_folder = osp.join(save_folder, 'images/input')
    util.mkdirs(save_folder)
    util.mkdirs(GT_folder)
    util.mkdirs(output_folder)
    util.mkdirs(input_folder)

    print('mkdir finish')

    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=False, tofile=True)
    logger = logging.getLogger('base')


    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}  # with border and center frames
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}  # with border and center frames
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        ms_ssim_rlt = {}  # with border and center frames
        ms_ssim_rlt_avg = {}
        ms_ssim_total_avg = 0.
        
        DB_ms_ssim_rlt = {}  # with border and center frames
        DB_ms_ssim_rlt_avg = {}
        DB_ms_ssim_total_avg = 0.

        bpp_rlt = {}
        bpp_rlt_avg = {}
        bpp_total_avg = 0.

        for val_data in val_loader:
            folder = val_data['folder'][0]
            idx_d = val_data['idx']
            LQ_basename = val_data['LQ_basename'][0]
            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []

            if ms_ssim_rlt.get(folder, None) is None:
                ms_ssim_rlt[folder] = []
                
            if DB_ms_ssim_rlt.get(folder, None) is None:
                DB_ms_ssim_rlt[folder] = []

            if bpp_rlt.get(folder, None) is None:
                bpp_rlt[folder] = []

            model.feed_data_runtest(val_data)
            if dataset_opt['calculate_real_bpp']:
                model.fullmodel_test()
            else:
                model.fullmodel_test_estimate()
            visuals = model.get_current_visuals()
            bpp = model.bpp
            # rlt_img = util.tensor2img(visuals['rlt'])  # uint8
            # gt_img = util.tensor2img(visuals['GT'])  # uint8
            rlt_img = util.tensor2img(pad.undo_pad(visuals['rlt'], *(model.cor_var_L)))  # uint8
            gt_img = util.tensor2img(pad.undo_pad(visuals['GT'], *(model.cor_real_H)))  # uint8
            input_img = util.tensor2img(pad.undo_pad(visuals['LQ'], *(model.cor_var_L)))
            # mid_ix = dataset_opt['N_frames'] // 2
            # input_img = util.tensor2img(visuals['LQ'][mid_ix])
            # input_img = util.tensor2img(visuals['LQ'])
            if save_imgs:
                try:
                    tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
                    LQ_tag = '{}.{}'.format(val_data['folder'], LQ_basename.replace('/', '-'))
                    print(osp.join(output_folder, '{}.png'.format(LQ_tag)))
                    cv2.imwrite(osp.join(output_folder, '{}.png'.format(LQ_tag)), rlt_img)
                    cv2.imwrite(osp.join(GT_folder, '{}.png'.format(LQ_tag)), gt_img)

                    cv2.imwrite(osp.join(input_folder, '{}.png'.format(LQ_tag)), input_img)


                except Exception as e:
                    print(e)
                    import ipdb; ipdb.set_trace()

            # calculate PSNR
            psnr = util.calculate_psnr(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)

            ssim = util.calculate_ssim(rlt_img, gt_img)
            ssim_rlt[folder].append(ssim)

            rlt_t = visuals['rlt'].unsqueeze_(dim=0)
            gt_t = visuals['GT'].unsqueeze_(dim=0)
            ms_ssim = util.calculate_ms_ssim(rlt_t, gt_t)
            ms_ssim_rlt[folder].append(ms_ssim)
            
            
            msssimDB = -10 * (torch.log(1 - ms_ssim) / np.log(10))
            DB_ms_ssim_rlt[folder].append(msssimDB)
            
            # bpp
            bpp_rlt[folder].append(bpp)

            tag = '{}.{}'.format(val_data['folder'], idx_d[0].replace('/', '-'))
            LQ_tag = '{}.{}'.format(val_data['folder'], LQ_basename.replace('/', '-'))
            logger.info(f'Image name:{LQ_tag}.png, BPP: {bpp:.4f}, PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, MS-SSIM:{ms_ssim:.4f}, MS-SSIM-DB:{msssimDB:.4f}')

            pbar.update('Test {} - {}'.format(folder, idx_d))
        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]

        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]

        for k, v in ms_ssim_rlt.items():
            ms_ssim_rlt_avg[k] = sum(v) / len(v)
            ms_ssim_total_avg += ms_ssim_rlt_avg[k]
            
            
        for k, v in DB_ms_ssim_rlt.items():
            DB_ms_ssim_rlt_avg[k] = sum(v) / len(v)
            DB_ms_ssim_total_avg += DB_ms_ssim_rlt_avg[k]
            

        for k, v in bpp_rlt.items():
            bpp_rlt_avg[k] = sum(v) / len(v)
            bpp_total_avg += bpp_rlt_avg[k]


        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        ms_ssim_total_avg /= len(ms_ssim_rlt)
        DB_ms_ssim_total_avg /= len(DB_ms_ssim_rlt)
        bpp_total_avg  /= len(bpp_rlt)

        log_s = '# Validation # PSNR: {:.4f}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4f}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # SSIM: {:.4f}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4f}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # MS-SSIM: {:.4f}:'.format(ms_ssim_total_avg)
        for k, v in ms_ssim_rlt_avg.items():
            log_s += ' {}: {:.4f}'.format(k, v)
        logger.info(log_s)
        
        
        log_s = '# Validation # MS-SSIM-dB: {:.4f}:'.format(DB_ms_ssim_total_avg)
        for k, v in DB_ms_ssim_rlt_avg.items():
            log_s += ' {}: {:.4f}'.format(k, v)
        logger.info(log_s)

        log_s = '# Validation # BPP: {:.4f}:'.format(bpp_total_avg)
        for k, v in bpp_rlt_avg.items():
            log_s += ' {}: {:.4f}'.format(k, v)
        logger.info(log_s)

        psnr_all = 0
        psnr_count = 0
        for k, v in psnr_rlt.items():
            psnr_all += sum(v)
            psnr_count += len(v)
        psnr_all = psnr_all * 1.0 / psnr_count
        psnr_out = "real Aver PSNR: {:.4f}".format(psnr_all)
        logger.info(psnr_out)

        ssim_all = 0
        ssim_count = 0
        for k, v in ssim_rlt.items():
            ssim_all += sum(v)
            ssim_count += len(v)
        ssim_all = ssim_all * 1.0 / ssim_count
        ssim_out = "real Aver SSIM: {:.4f}".format(ssim_all)
        logger.info(ssim_out)

        ms_ssim_all = 0
        ms_ssim_count = 0
        for k, v in ms_ssim_rlt.items():
            ms_ssim_all += sum(v)
            ms_ssim_count += len(v)
        ms_ssim_all = ms_ssim_all * 1.0 / ms_ssim_count
        ms_ssim_out = "real Aver MS-SSIM: {:.4f}".format(ms_ssim_all)
        logger.info(ms_ssim_out)
        
        DB_ms_ssim_all = 0
        DB_ms_ssim_count = 0
        for k, v in DB_ms_ssim_rlt.items():
            DB_ms_ssim_all += sum(v)
            DB_ms_ssim_count += len(v)
        DB_ms_ssim_all = DB_ms_ssim_all * 1.0 / DB_ms_ssim_count
        DB_ms_ssim_out = "real Aver MS-SSIM-DB: {:.4f}".format(DB_ms_ssim_all)
        logger.info(DB_ms_ssim_out)

        bpp_all = 0
        bpp_count = 0
        for k, v in bpp_rlt.items():
            bpp_all += sum(v)
            bpp_count += len(v)
        bpp_all = bpp_all * 1.0 / bpp_count
        bpp_out = "real Aver BPP: {:.4f}".format(bpp_all)
        logger.info(bpp_out)

if __name__ == '__main__':
    main()

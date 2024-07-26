import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from utils.util import AverageMeter, Metrics
from data import create_dataloader, create_dataset
from data import pretrain_dataset
from models import create_model
import numpy as np
import cv2
from tqdm import tqdm
from models import pad


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='./options/train/all_data_combine_psnr.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        world_size = 1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None or not os.path.exists(opt['path']['experiments_root']):
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=False, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            # tb_logger_foldername = opt['name'] + '_' + '{}'.format(opt['train']['lambdas']['lambda_metric']) + '_' + '{}'.format(opt['train']['lambdas']['lambda_fea'])
            tb_logger_foldername = opt['name'] \
                                   + '_' + 'lmetric_{}'.format(opt['train']['lambdas']['lambda_metric']) \
                                   + '_' + 'lfea_{}'.format(opt['train']['lambdas']['lambda_fea']) \
                                   + '_' + 'lvgg_{}'.format(opt['train']['lambdas']['lambda_vgg']) \
                                   + '_' + 'lbpp_{}'.format(opt['train']['lambdas']['lambda_bpp'])
            tb_logger = SummaryWriter(log_dir='./tb_logger/' + tb_logger_foldername)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    total_epochs = 0
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            total_train_set = create_dataset(dataset_opt)
            total_train_size = int(math.ceil(len(total_train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / total_train_size))
            if opt['dist']:
                total_train_sampler = DistIterSampler(total_train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (total_train_size * dataset_ratio)))
            else:
                total_train_sampler = None
            total_train_loader = create_dataloader(total_train_set, dataset_opt, opt, total_train_sampler)
            if rank <= 0:
                logger.info('Number of total_train images: {:,d}, iters: {:,d}'.format(
                    len(total_train_set), total_train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
            assert total_train_loader is not None
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            # val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))


    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        if opt['train']['finetune']:
            loss_best = 1e10
        else:
            loss_best = resume_state['loss']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        del resume_state
    else:
        current_step = 0
        start_epoch = 0
        loss_best = 1e10

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    pretrain_iters = opt['train']['pretrain_iters']
    comrpess_flag = opt['train']['compress']

    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            total_train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(total_train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            if comrpess_flag and current_step >= pretrain_iters:
                compress = True
            else:
                compress = False
            model.feed_data(train_data)
            model.optimize_parameters(current_step, compress=compress)
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'], compress=compress)
            #### log
            logs = model.get_current_log()
            if current_step % opt['logger']['print_freq'] == 0:
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '||{:s}: {:.4f} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            if rank <= 0 and logs['loss'] < loss_best and compress:
                loss_best = logs['loss']
                logger.info(f'******************iters: {current_step}, loss: {loss_best.item():.6f} Best!*****************')
                model_save_name = f"best_{current_step}"
                model.save(model_save_name)
            calculate_Metrics = Metrics()
            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['dist']:
                    # multi-GPU testing
                    psnr_rlt = {}  # with border and center frames
                    bpp_rlt = {}
                    ms_ssim_rlt = {}
                    if rank == 0:
                        pbar = util.ProgressBar(len(val_set))
                        pbar.start()
                    # temp_cnt = 0
                    for idx in range(rank, len(val_set), world_size):
                        # temp_cnt += 1
                        # if temp_cnt > 2:
                        #     break
                        val_data = val_set[idx]
                        val_data['LQs'].unsqueeze_(0)
                        val_data['GT'].unsqueeze_(0)
                        folder = val_data['folder']
                        idx_d, max_idx = val_data['idx'].split('/')
                        idx_d, max_idx = int(idx_d), int(max_idx)
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                           device='cuda')
                        if bpp_rlt.get(folder, None) is None:
                            bpp_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                           device='cuda')
                        if ms_ssim_rlt.get(folder, None) is None:
                            ms_ssim_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                           device='cuda')
                        # tmp = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                        model.feed_data(val_data)
                        net_output = model.fullmodel_val(compress=compress)
                        visuals = model.get_current_visuals()
                        sou_img = util.tensor2img(visuals['LQ'])
                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        rlt_img2 = util.tensor2img(visuals['rlt2'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8
                        ill_img = util.tensor2img(visuals['ill'])
                        rlt_img3 = util.tensor2img(visuals['rlt3'])

                        save_img = np.concatenate([sou_img, rlt_img, ill_img, gt_img, rlt_img3, rlt_img2], axis=0)
                        im_path = os.path.join(opt['path']['val_images'], '%06d.png' % current_step)
                        cv2.imwrite(im_path, save_img.astype(np.uint8))

                        # calculate PSNR
                        temp_val_data_GT = val_data['GT'].to(device='cuda')
                        bpp, psnr, ms_ssim = calculate_Metrics(net_output, temp_val_data_GT, compress=compress)
                        psnr_rlt[folder][idx_d] = psnr
                        bpp_rlt[folder][idx_d] = bpp
                        ms_ssim_rlt[folder][idx_d] = ms_ssim
                    if rank == 0:
                        pbar.update('Val processing finish!')
                    # # collect data
                    for _, v in psnr_rlt.items():
                        dist.reduce(v, 0)
                    for _, v in bpp_rlt.items():
                        dist.reduce(v, 0)
                    for _, v in ms_ssim_rlt.items():
                        dist.reduce(v, 0)
                    dist.barrier()

                    if rank == 0:
                        bpp_rlt_avg = {}
                        bpp_total_avg = 0.
                        for k, v in bpp_rlt.items():
                            bpp_rlt_avg[k] = torch.mean(v).cpu().item()
                            bpp_total_avg += bpp_rlt_avg[k]
                        bpp_total_avg /= len(bpp_rlt)
                        log_s = '# Validation # BPP: {:.4f}:'.format(bpp_total_avg)
                        for k, v in bpp_rlt_avg.items():
                            log_s += ' {}: {:.4f}'.format(k, v)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('bpp_avg', bpp_total_avg, current_step)
                            for k, v in bpp_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)

                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                            psnr_total_avg += psnr_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        log_s = '# Validation # PSNR: {:.4f}:'.format(psnr_total_avg)
                        for k, v in psnr_rlt_avg.items():
                            log_s += ' {}: {:.4f}'.format(k, v)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)

                        ms_ssim_rlt_avg = {}
                        ms_ssim_total_avg = 0.
                        for k, v in ms_ssim_rlt.items():
                            ms_ssim_rlt_avg[k] = torch.mean(v).cpu().item()
                            ms_ssim_total_avg += ms_ssim_rlt_avg[k]
                        ms_ssim_total_avg /= len(ms_ssim_rlt)
                        log_s = '# Validation # MS-SSIM: {:.4f}:'.format(ms_ssim_total_avg)
                        for k, v in ms_ssim_rlt_avg.items():
                            log_s += ' {}: {:.4f}'.format(k, v)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('ms_ssim_avg', ms_ssim_total_avg, current_step)
                            for k, v in ms_ssim_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
                else:
                    logger.info('skip val')

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step, loss_best)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierLoss2
from models.criterion import Criterion
import time

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, find_unused_parameters=True, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            # loss_type = train_opt['pixel_criterion']
            # if loss_type == 'l1':
            #     self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            # elif loss_type == 'l2':
            #     self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            # elif loss_type == 'cb':
            #     self.cri_pix = CharbonnierLoss().to(self.device)
            # elif loss_type == 'cb2':
            #     self.cri_pix = CharbonnierLoss2().to(self.device)
            # else:
            #     raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            # self.l_pix_w = train_opt['pixel_weight']
            #
            # self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                aux_parameters = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if k.endswith(".quantiles"):
                            aux_parameters.append(v)
                        else:
                            optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                # Make sure we don't have an intersection of parameters
                params_dict = dict(self.netG.named_parameters())
                inter_params = set(optim_params) & set(aux_parameters)
                union_params = set(optim_params) | set(aux_parameters)
                assert len(inter_params) == 0
                assert len(union_params) - len(params_dict.keys()) == 0

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_aux_par = torch.optim.Adam(aux_parameters,
                                                lr=train_opt['lr_aux'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_aux_par)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            elif train_opt['lr_scheme'] == 'LambdaLR':
                warm_up_counts = train_opt['warm_up_counts']
                gamma = train_opt['lr_gamma']
                milestones = train_opt['milestones']
                warm_up_with_multistep_lr = lambda i: (i + 1) / warm_up_counts if i < warm_up_counts else gamma ** len([m for m in milestones if m <= i])
                self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(self.optimizers[0], lr_lambda=warm_up_with_multistep_lr, verbose=False))
                warm_up_with_multistep_lr = lambda i: (i + 1) / warm_up_counts if i < warm_up_counts else 1.0
                self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(self.optimizers[1], lr_lambda=warm_up_with_multistep_lr))
            elif train_opt['lr_scheme'] == 'MultiStepLR':
                gamma = train_opt['lr_gamma']
                milestones = train_opt['milestones']
                milestones_aux = train_opt['milestones_aux']
                self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(self.optimizers[0],
                                                                            milestones=milestones,
                                                                            gamma=gamma))
                self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(self.optimizers[1],
                                                                            milestones=milestones_aux,
                                                                            gamma=0.5))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()
            self.loss = Criterion(opt)

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf = data['nf'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        """
        ************************************************************
        input: step
        output:
            renconstruct image: x_hat,
            lowlight compress feature1: y_inter,
            gt compress feature1: y_inter_gt,
            compress feature: y,
            gt compress feature: y_gt,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        ************************************************************
        """
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max+0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()

        net_output = self.netG(self.var_L, gt=self.real_H, mask=mask, ll_enhance=True)  # x, gt=None, mask, ll_enhance
        loss = self.loss(net_output, self.real_H)

        final_loss = loss['loss']
        self.log_dict['loss'] = loss['loss']
        # do optimization if and only if the loss is small (rec is somehow bounded with 0-1)
        optimizer_flag = loss["loss"].item() >= 0 and loss["loss"].item() < self.opt['train']['loss_cap']
        if not optimizer_flag:
            message = '[Warning]: network parameters are not optimized due to train loss = {:.4f}.'.format(loss['loss'].item())
            logger.info(message)
        if self.opt['train']['clip_max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt['train']['clip_max_norm'])

        final_loss.backward()
        if not optimizer_flag:
            self.optimizers[0].zero_grad()
        self.optimizers[0].step()
        # aux_optimizer
        aux_loss = self.netG.module.aux_loss()
        self.log_dict['aux_loss'] = aux_loss
        aux_loss.backward()
        if not optimizer_flag:
            self.optimizers[1].zero_grad()
        self.optimizers[1].step()

    def optimize_pretrain_parameters(self, data):
        """
        ************************************************************
        input: source image
        output:
            renconstruct image: x_hat,
            lowlight compress feature1: y_inter,
            gt compress feature1: y_inter_gt,
            compress feature: y,
            gt compress feature: y_gt,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        ************************************************************
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        data = data.to(self.device)
        net_output = self.netG(data)
        loss = self.loss(net_output, data)

        pretrain_loss = loss['loss']
        self.log_dict['loss'] = loss['loss']
        # do optimization if and only if the loss is small (rec is somehow bounded with 0-1)
        optimizer_flag = loss["loss"].item() >= 0 and loss["loss"].item() < self.opt['train']['loss_cap']
        if not optimizer_flag:
            message = '[Warning]: network parameters are not optimized due to train loss = {:.4f}.'.format(loss['loss'].item())
            logger.info(message)
        if self.opt['train']['clip_max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.opt['train']['clip_max_norm'])
        pretrain_loss.backward()
        if not optimizer_flag:
            self.optimizers[0].zero_grad()
        self.optimizers[0].step()
        # aux_optimizer
        aux_loss = self.netG.module.aux_loss()
        self.log_dict['aux_loss'] = aux_loss
        aux_loss.backward()
        if not optimizer_flag:
            self.optimizers[1].zero_grad()
        self.optimizers[1].step()

    def pretrain_val(self, data):
        with torch.no_grad():
            x = data.to(self.device)
            net_output = self.netG(x, gt=None, mask=None, ll_enhance=False)
        return net_output

    def pretrain_test(self, data):
        x = data.to(self.device)
        bitstreams = self.netG.module.compress(x, mask=None, ll_enhance=False)
        x_hat = self.netG.module.decompress(self, bitstreams['strings'], bitstreams['shape'])
        return x_hat

    def fullmodel_val(self):
        self.netG.eval()
        with torch.no_grad():
            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()
            # self.fake_H = self.netG(self.var_L, mask)
            net_output = self.netG(self.var_L, gt=self.real_H, mask=mask, ll_enhance=True)  # x, gt=None, mask, ll_enhance
            self.fake_H = net_output['x_hat']
        self.netG.train()
        return net_output

    def fullmodel_test(self):
        self.netG.eval()
        with torch.no_grad():
            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            bitstreams = self.netG.module.compress(self.var_L, mask=mask, ll_enhance=True)
            shape_bitstream = (bitstreams['shape'][0], bitstreams['shape'][1])
            x_hat = self.netG.module.decompress(self, bitstreams['strings'], shape_bitstream)
            self.fake_H = x_hat['x_hat']
            num_pixels = dark.size(0) * dark.size(2) * dark.size(3)
            bpp = sum(len(s[0]) for s in bitstreams["strings"]) * 8.0 / num_pixels
            self.bpp = bpp

    def fullmodel_test_estimate(self):
        self.netG.eval()
        with torch.no_grad():
            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()
            # self.fake_H = self.netG(self.var_L, mask)
            net_output = self.netG(self.var_L, gt=self.real_H, mask=mask, ll_enhance=True)  # x, gt=None, mask, ll_enhance
            self.fake_H = net_output['x_hat']
            num_pixels = dark.size(0) * dark.size(2) * dark.size(3)
            bpp = sum((-torch.log2(likelihoods).sum() / num_pixels) for likelihoods in net_output['likelihoods'].values())
            self.bpp = bpp.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max+0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()
            self.fake_H = self.netG(self.var_L, mask)
        self.netG.train()

    def test4(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 400
            W_new = 608
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()


    def test5(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 384
            W_new = 384
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        if not(len(self.nf.shape) == 4):
            self.nf = self.nf.unsqueeze(dim=0)
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        mask = mask.repeat(1, 3, 1, 1)
        out_dict['rlt3'] = mask[0].float().cpu()

        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['ill'] = mask[0].float().cpu()
        out_dict['rlt2'] = self.nf.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del dark
        del light
        del mask
        del noise
        del self.real_H
        del self.nf
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

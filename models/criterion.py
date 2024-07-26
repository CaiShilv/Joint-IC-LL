import math
import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.opt = opt
        # criterions
        self.criterion_metric = opt['train']['criterions']['criterion_metric']
        self.criterion_fea = opt['train']['criterions']['criterion_fea']

        # lambdas
        self.lambda_metric = opt['train']['lambdas']['lambda_metric']
        self.lambda_fea = opt['train']['lambdas']['lambda_fea']
        self.lambda_bpp = opt['train']['lambdas']['lambda_bpp']
        self.lambda_vgg = opt['train']['lambdas']['lambda_vgg']
        self.metric_loss = RateDistortionLoss(lmbda=self.lambda_metric, criterion=self.criterion_metric)
        if self.lambda_vgg > 0.:
            self.perception_loss = VGGLoss()
        if self.criterion_fea:
            self.fea_loss = FeaLoss(lmbda=self.lambda_fea, criterion=self.criterion_fea)

    def forward(self, out_net, gt, compress=True):
        out = {'loss': torch.tensor(0.).cuda(),
               'weight_fea_loss': torch.tensor(0.).cuda(),
               'weight_dis_loss': torch.tensor(0.).cuda(),
               'fea_loss': torch.tensor(0.).cuda(),
               'dis_loss': torch.tensor(0.).cuda(),
               'bpp_loss': torch.tensor(0.).cuda(),
               'weight_bpp_loss': torch.tensor(0.).cuda(),
               'weight_vgg_loss': torch.tensor(0.).cuda()
               }
        # bpp loss and metric loss
        out_metric = self.metric_loss(out_net, gt, compress=compress)
        if compress:
            out['weight_bpp_loss'] = self.lambda_bpp * out_metric['bpp_loss']
            out['loss'] += out['weight_bpp_loss']
            out['bpp_loss'] = out_metric['bpp_loss']

        if self.criterion_metric == 'mse':
            out['dis_loss'] = out_metric['mse_loss']
        elif self.criterion_metric == 'ms-ssim':
            out['dis_loss'] = out_metric['ms_ssim_loss']
        elif self.criterion_metric == 'cb':
            out['dis_loss'] = out_metric['dis_loss']
        elif self.criterion_metric == 'l1':
            out['dis_loss'] = out_metric['dis_loss']

        # calculate dis_loss
        for k, v in out_metric.items():
            out[k] = v
            if 'weighted' in k:
                out['loss'] += v
                out['weight_dis_loss'] = v

        # perception_loss
        if self.lambda_vgg > 0.:
            l_vgg = self.lambda_vgg * self.perception_loss(out_net["x_hat"], gt)
            out['weight_vgg_loss'] = l_vgg
            out['loss'] += l_vgg

        # fea loss
        if self.criterion_fea and out_net['y_inter_gt'] is not None:
            if 'y_inter' in out_net.keys():
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'], out_net['y_inter'], out_net['y_inter_gt'])
            else:
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'])
            out['fea_loss'] = out_fea['fea_loss']
            for k, v in out_fea.items():
                out[k] = v
                if 'weighted' in k:
                    out['loss'] += v
                    out['weight_fea_loss'] = v
        return out


# rate distortion loss
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2, criterion='mse'):
        super().__init__()
        self.lmbda = lmbda
        self.criterion = criterion
        if self.criterion == 'mse':
            self.loss = nn.MSELoss()
        elif self.criterion == 'ms-ssim':
            from pytorch_msssim import ms_ssim
            self.loss = ms_ssim
        elif self.criterion == 'cb':
            self.loss = CharbonnierLoss2()
        elif self.criterion == 'l1':
            self.loss = nn.L1Loss()
        else:
            NotImplementedError('RateDistortionLoss criterion [{:s}] is not recognized.'.format(criterion))

    def forward(self, out_net, target, compress=True):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        if compress:
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )
        else:
            out["bpp_loss"] = 0
        if self.criterion == 'mse':
            out["mse_loss"] = self.loss(out_net["x_hat"], target)
            out["weighted_dis_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"]
        elif self.criterion == 'ms-ssim':
            out["ms_ssim_loss"] = 1 - self.loss(out_net["x_hat"], target, data_range=1.0)
            out["weighted_dis_loss"] = self.lmbda * out["ms_ssim_loss"]
        elif self.criterion == 'cb':
            out["dis_loss"] = self.loss(out_net["x_hat"], target)
            out["weighted_dis_loss"] = self.lmbda * out["dis_loss"]
        elif self.criterion == 'l1':
            out["dis_loss"] = self.loss(out_net["x_hat"], target)
            out["weighted_dis_loss"] = self.lmbda * 255 ** 2 * out["dis_loss"]
        return out


# fea loss
class FeaLoss(nn.Module):
    def __init__(self, lmbda=1., criterion='l2'):
        super(FeaLoss, self).__init__()
        self.lmbda = lmbda
        self.criterion = criterion
        if self.criterion == 'l2':
            self.loss = nn.MSELoss()
        elif self.criterion == 'l1':
            self.loss = nn.L1Loss()
        elif self.criterion == 'cb':
            self.loss = CharbonnierLoss2()
        else:
            NotImplementedError('FeaLoss criterion [{:s}] is not recognized.'.format(criterion))

    def forward(self, fea, fea_gt, fea_inter=None, fea_inter_gt=None):
        loss = self.loss(fea, fea_gt)
        if fea_inter is not None and fea_inter_gt is not None:
            loss += self.loss(fea_inter, fea_inter_gt)
        out = {
            'fea_loss': loss,
            'weighted_fea_loss': loss * self.lmbda,
        }
        return out

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward1(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss
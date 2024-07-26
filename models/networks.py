import torch
# import models.archs.low_light_transformer as low_light_transformer
import models.archs.low_light_transformer as Joint_compression_low_light_transformer
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    if which_model == 'low_light_transformer':
        netG = Joint_compression_low_light_transformer.joint_compression_low_light_v1(N=opt_net['nf'], front_RBs=opt_net['front_RBs'], local_RBs=opt_net['local_RBs'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


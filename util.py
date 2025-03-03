import argparse
import torch
import torch.nn as nn


def str2bool(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(str):
    try:
        return [int(i) for i in str.split(',')]
    except ValueError:
        raise ValueError


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


def load_pretrained_style_generator(path, generator, stns, rt_stns):
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt['g'], strict=True)
    if stns is not None:
        stns.load_state_dict(ckpt['stns'], strict=True)
    if rt_stns is not None:
        rt_stns.load_state_dict(ckpt['rtstn'], strict=True)
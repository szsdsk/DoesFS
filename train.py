import argparse
import math
import torch
import warnings
import os
from tqdm import tqdm
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.backends import cudnn
from torchvision import transforms
from copy import deepcopy
from PIL import Image
from splice_utils.splice import Splice
from lpips_loss.modules.lpips import LPIPS
from model import DeformAwareGenerator, DiscriminatorPatch, Extra, TPSSpatialTransformer, RTSpatialTransformer
from util import str2bool, str2list, accumulate, requires_grad

warnings.filterwarnings('ignore')
cudnn.benchmark = True
torch.manual_seed(3202)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturation_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
                (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def warp_reg_loss(warp_flow):
    dx_reg = 1 - F.cosine_similarity(warp_flow[:, :, :-1, :-1], warp_flow[:, :, :-1, 1:])
    dy_reg = 1 - F.cosine_similarity(warp_flow[:, :, :-1, :-1], warp_flow[:, :, 1:, :-1])
    return (dx_reg + dy_reg).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--style', type=str, default='style1')
    parser.add_argument('--source', type=str, default='source1.png')
    parser.add_argument('--target', type=str, default='target1.png')
    parser.add_argument('--style_ref', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='checkpoints/stylegan2-ffhq-config-f.pt')
    parser.add_argument('--lpips_dir', type=str, default='checkpoints', help='location of lpips_loss models. Used alex')
    parser.add_argument('--img_res', type=int, default=1024)
    parser.add_argument('--num_iter', type=int, default=500)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--warp_res', type=str, default='32,64')
    parser.add_argument('--warp_gs', type=str, default='10,10')
    parser.add_argument('--cross_mode', type=str, default='f')
    parser.add_argument('--within_mode', type=str, default='f')
    parser.add_argument('--cross_layers', type=str, default='5,11')
    parser.add_argument('--within_layers', type=str, default='5')

    parser.add_argument('--g_lr', type=float, default=2e-3)
    parser.add_argument('--d_lr', type=float, default=2e-3)
    parser.add_argument('--stn_lr', type=float, default=5e-6)
    parser.add_argument('--rtstn_lr', type=float, default=1e-4)
    parser.add_argument('--adv_wt', type=float, default=1)
    parser.add_argument('--a2agg_wt', type=float, default=50000.)
    parser.add_argument('--a2agr_wt', type=float, default=50000.)
    parser.add_argument('--a2b_wt', type=float, default=6)
    parser.add_argument('--warp_wt', type=float, default=1e-6)
    parser.add_argument('--use_stn', type=str2bool, default=True)
    parser.add_argument('--use_rtstn', type=str2bool, default=True)
    parser.add_argument('--tune_g', type=str2bool, default=True)
    parser.add_argument('--stn_accum', type=float, default=0.995)
    parser.add_argument('--g_accum', type=float, default=0.5 ** (32 / (10 * 1000)))
    parser.add_argument('--hp', type=int, default=1)
    parser.add_argument('--swap_layer', type=int, default=8)

    args = parser.parse_args()
    assert args.use_stn or args.tune_g, 'at least one of these two args to be `True`'

    device = args.device
    hp = args.hp
    num_iter = args.num_iter  # 500
    # 模型权重相加
    g_accum = args.g_accum
    stn_accum = args.stn_accum
    rt_warp_resolutions = tps_warp_resolutions = str2list(args.warp_res)
    warp_grid_sizes = str2list(args.warp_gs)
    latent_dim = 512
    mean_patch_length = 0

    transform = transforms.Compose([transforms.Resize((args.img_res, args.img_res)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 初始化 generator and discriminator
    original_generator = DeformAwareGenerator(args.img_res, latent_dim, 8, 2, resolutions=tps_warp_resolutions,
                                              rt_resolutions=rt_warp_resolutions).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    original_generator.load_state_dict(ckpt["g_ema"], strict=True)
    generator = deepcopy(original_generator).eval()

    g_ema = deepcopy(original_generator).eval()
    g_module = generator
    accumulate(g_ema, generator, 0)

    discriminator = DiscriminatorPatch(args.img_res).to(device).eval()
    discriminator.load_state_dict(ckpt['d'], strict=True)

    # patch-level adversarial loss
    extra = Extra().to(device)

    # deformation modules
    stns = TPSSpatialTransformer(resolutions=tps_warp_resolutions, grid_size=warp_grid_sizes).to(device)
    rt_stns = RTSpatialTransformer(resolutions=rt_warp_resolutions).to(device)
    stns_ema = TPSSpatialTransformer(resolutions=tps_warp_resolutions, grid_size=warp_grid_sizes).to(device)
    rt_stns_ema = RTSpatialTransformer(resolutions=rt_warp_resolutions).to(device)

    # Dino feature extractor
    splice = Splice(device=device)

    softmax = nn.Softmax(dim=0)
    # 1 * 1 * 512 => 1 * 18 * 512
    mean_latent = original_generator.mean_latent(1).unsqueeze(0).repeat(1, original_generator.n_latent, 1)
    # (8, 18)
    swap = [i for i in range(args.swap_layer, original_generator.n_latent)]

    # load image (aligned)
    style_path = os.path.join('data/style_images_aligned', args.target)
    style_aligned = Image.open(style_path).convert('RGB')
    style_image = transform(style_aligned).to(device).unsqueeze(0)

    real_path = os.path.join('data/style_images_aligned', args.source)
    real_aligned = Image.open(real_path).convert('RGB')
    real_image = transform(real_aligned).to(device).unsqueeze(0)

    if args.style_ref is None:
        args.style_ref = args.target
    ref_path = os.path.join('data/style_images_aligned', args.style_ref)
    ref_aligned = Image.open(ref_path).convert('RGB')
    ref_image = transform(ref_aligned).to(device).unsqueeze(0)

    # initialize optimizers
    params_d = []
    if args.tune_g:
        params_d.append({'params': generator.parameters(), 'lr': args.g_lr})
    if args.use_stn:
        params_d.append({'params': stns.parameters(), 'lr': args.stn_lr})
    if args.use_rtstn:
        params_d.append({'params': rt_stns.parameters(), 'lr': args.rtstn_lr})

    g_optim = optim.Adam(params_d, betas=(.1, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0, 0.99))
    e_optim = optim.Adam(extra.parameters(), lr=args.d_lr, betas=(0, 0.99))

    mode_cross = args.cross_mode
    mode_within = args.within_mode
    vit_layer_id_cross = str2list(args.cross_layers)
    vit_layer_id_within = str2list(args.within_layers)

    # inverse source reference for color alignment
    w_plus_src = mean_latent.clone()
    w_plus_src.requires_grad_(True)
    params = [{'params': w_plus_src, 'lr': 2e-3}]
    optimizer = optim.Adam(params)
    loss_lpips = LPIPS(model_dir=args.lpips_dir).to(device)

    pbar = tqdm(range(300))
    for idx in pbar:
        # 用该latent_code生成图片
        Gw, _ = original_generator(w_plus_src, input_is_latent=True)
        l1_loss = F.l1_loss(Gw, real_image)
        lpips_loss = loss_lpips(Gw, real_image).mean()
        loss = l1_loss + lpips_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    exp_latent_src = w_plus_src.clone()

    # inverse target reference for color alignment
    w_plus_tgt = mean_latent.clone()
    w_plus_tgt.requires_grad_(True)
    params = [{'params': w_plus_tgt, 'lr': 2e-3}]
    optimizer = optim.Adam(params)
    loss_lpips = LPIPS(model_dir=args.lpips_dir).to(device)

    pbar = tqdm(range(300))
    for idx in pbar:
        Gw, _ = original_generator(w_plus_tgt, input_is_latent=True)
        l1_loss = F.l1_loss(Gw, style_image)
        lpips_loss = loss_lpips(Gw, style_image).mean()
        loss = l1_loss + lpips_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    exp_latent_tgt = w_plus_tgt.clone()

    del loss_lpips, optimizer








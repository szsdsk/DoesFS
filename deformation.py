import argparse
import os
from copy import deepcopy
import tqdm
import torch
import torchvision
from face_align import align_face
import torchvision.transforms as transforms
from model import RTSpatialTransformer, TPSSpatialTransformer, DeformAwareGenerator
from util import str2list, str2bool, load_pretrained_style_generator
from PIL import Image
from e4e_projection import projection as e4e_projection
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate deformed stylized faces with different alpha.")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--style', type=str, default='style1')
    parser.add_argument('--use_stn', type=str2bool, default=True)
    parser.add_argument('--use_rtstn', type=str2bool, default=True)
    parser.add_argument('--n_sample', type=int, default=1)
    parser.add_argument('--input_image', type=str, default='./data/test_inputs/001.png')
    parser.add_argument('--nrow', type=int, default=1)
    parser.add_argument('--swap_res', type=str, default='32,64')
    parser.add_argument('--swap_gs', type=str, default='10,10')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--alpha0', type=float, default=-0.5)
    parser.add_argument('--alpha1', type=float, default=1)
    parser.add_argument('--split', type=int, default=5)
    parser.add_argument('--align', type=str2bool, default=True, help='set as True if input image is not aligned.')

    args = parser.parse_args()

    device = args.device
    args.nrow = args.split + 1
    n_sample = args.n_sample
    style = args.style
    rt_swap_resolutions = tps_swap_resolutions = str2list(args.swap_res)
    swap_grid_sizes = str2list(args.swap_gs)
    output_dir = f'outputs/control' if args.output is None else args.output
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load input image
    input_image = Image.open(args.input_image).convert('RGB')
    # face align
    if args.align:
        input_image = align_face(args.input_image)

    # e4e inversion
    latent_path = os.path.join(output_dir, 'code_' + os.path.basename(args.input_image).split('.')[0] + '.pt')
    if not os.path.exists(latent_path):
        latent = e4e_projection(input_image, latent_path, device)
    else:
        latent = torch.load(latent_path)['latent']

    latent = latent.unsqueeze(0)

    # Load generator
    original_generator = DeformAwareGenerator(1024, 512, 8, 2, resolutions=tps_swap_resolutions,
                                              rt_resolutions=rt_swap_resolutions).to(device)
    ckpt = torch.load('checkpoints/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=True)

    generator = deepcopy(original_generator)
    stns = TPSSpatialTransformer(grid_size=swap_grid_sizes, resolutions=tps_swap_resolutions).to(device)
    rt_stns = RTSpatialTransformer(resolutions=rt_swap_resolutions).to(device)

    ckpt = f'./checkpoints/{args.style}.pt'
    print(f'Loading deformable generator from checkpoint: {ckpt}')
    load_pretrained_style_generator(ckpt, generator, stns, rt_stns)

    step = (args.alpha1 - args.alpha0) / args.split
    alphas = [args.alpha0 + step * i for i in range(args.split + 1)]

    with torch.no_grad():
        generator.eval()

        controls = []
        for alpha in alphas:
            # inference
            result, _ = generator(latent, input_is_latent=True, stns=stns, rt_stns=rt_stns, alpha=alpha)
            controls.append(result)

        input_tensor = transform(input_image).unsqueeze(0).to(device)

        results = [result.clamp(-1, 1) for result in controls]
        image_tensors = [input_tensor] + results

        concatenated = torch.cat(image_tensors, dim=3)

        torchvision.utils.save_image(concatenated,
                                     f'{output_dir}/{args.style}.png',
                                     normalize=True, value_range=(-1, 1), nrow=args.nrow+1)

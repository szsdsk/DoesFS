import argparse
import os
import torch
import torchvision.utils as utils
from PIL import Image
from face_align import align_face
from torchvision import transforms
from model import DeformAwareGenerator, TPSSpatialTransformer, RTSpatialTransformer
from util import str2bool, str2list, load_pretrained_style_generator
from e4e_projection import projection as e4e_projection
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Generate deformed stylized faces with different alpha.")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--swap_res', type=str, default='32,64')
parser.add_argument('--swap_gs', type=str, default='10,10')
parser.add_argument('--input_image', type=str, default='./data/test_inputs/001.png')
parser.add_argument('--align', type=str2bool, default=True, help='set as True if input image is not aligned.')
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--alpha', type=float, default=0.8, help='deformation degree')
args = parser.parse_args()

device = args.device
rt_swap_resolutions = tps_swap_resolutions = str2list(args.swap_res)
swap_grid_sizes = str2list(args.swap_gs)
output_dir = f'outputs/inference' if args.output is None else args.output
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

results = []
for i in range(1, 8):
    # load deformable generator
    generator = DeformAwareGenerator(1024, 512, 8, 2, resolutions=tps_swap_resolutions,
                                     rt_resolutions=rt_swap_resolutions).to(device)
    stns = TPSSpatialTransformer(grid_size=swap_grid_sizes, resolutions=tps_swap_resolutions).to(device)
    rt_stns = RTSpatialTransformer(resolutions=rt_swap_resolutions).to(device)
    load_pretrained_style_generator(f'./checkpoints/style{i}.pt', generator, stns, rt_stns)
    generator.eval()

    # inference
    result, _ = generator(latent, input_is_latent=True, stns=stns, rt_stns=rt_stns, alpha=args.alpha)
    results.append(result)

input_tensor = transform(input_image).unsqueeze(0).to(device)

results = [result.clamp(-1, 1) for result in results]
image_tensors = [input_tensor] + results

concatenated = torch.cat(image_tensors, dim=3)

output_path = os.path.join(output_dir, f'styled_{os.path.basename(args.input_image)}')
utils.save_image(concatenated, output_path, normalize=True, value_range=(-1, 1))
print(f"Concatenated image saved at: {output_path}")

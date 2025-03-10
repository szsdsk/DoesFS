import os

from splice_utils.splice import Splice
from torchvision import transforms
import torch
from argparse import ArgumentParser
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize(args):
    # load the image
    input_img = Image.open(args.image_path).convert('RGB')
    input_img = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])(input_img).unsqueeze(0).to(device)

    # define the extractor
    dino_preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    vit_extractor = Splice(device)
    image = dino_preprocess(input_img)

    # calculate
    with torch.no_grad():
        self_sim = vit_extractor.get_ssim[args.mode](image, args.layer)

    pca = PCA(n_components=3)
    self_sim_cpu = self_sim[0].cpu().numpy()
    pca.fit(self_sim_cpu)
    reduced = pca.transform(self_sim_cpu)[None, ...]

    # reshape the reduced keys to the image shape
    patch_size = vit_extractor.extractor.get_patch_size() - 1
    patch_h_num = vit_extractor.extractor.get_height_patch_num(input_img.shape)
    patch_w_num = vit_extractor.extractor.get_width_patch_num(input_img.shape)
    pca_image = reduced.reshape(patch_h_num, patch_w_num, 3)
    pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
    h, w, _ = pca_image.shape
    pca_image = Image.fromarray(np.uint8(pca_image * 255))
    pca_image = transforms.Resize((h * patch_size, w * patch_size), interpolation=transforms.InterpolationMode.BILINEAR)(pca_image)
    pca_image.save(args.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, default="source1.png")
    parser.add_argument("--image_path", type=str, default='data/style_images_aligned/')
    parser.add_argument("--layer", type=int, default=11,
                        help='Transformer layer from which to extract the feature, between 0-11')
    parser.add_argument("--save_path", type=str, default='outputs/pca/')
    parser.add_argument("--mode", type=str, default='k')
    args = parser.parse_args()
    args.layer -= 1
    os.makedirs(args.save_path, exist_ok=True)
    args.image_path += args.image
    args.save_path += str(args.layer + 1) + args.mode + "_" + args.image
    visualize(args)
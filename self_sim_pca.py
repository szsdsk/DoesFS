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
    # with torch.no_grad():
    #     self_sim = vit_extractor.get_ssim[args.mode](image, args.layer)
    pca_images = []
    for layer in [3, 6, 12]:
        layer -= 1
        for mode in 'kf':
            with torch.no_grad():
                self_sim = vit_extractor.get_ssim[mode](image, layer)
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
            # pca_image.save(args.save_path)
            pca_images.append(pca_image)

    # 拼接图像成 3x2 网格
    width, height = pca_images[0].size
    grid_width = width * 2  # 2列
    grid_height = height * 3  # 3行

    grid_image = Image.new('RGB', (grid_width, grid_height))
    # 将六张图像按照 3x2 布局粘贴到网格中
    for idx, img in enumerate(pca_images):
        row = idx // 2  # 计算行号 (0, 1, 2)
        col = idx % 2  # 计算列号 (0, 1)
        grid_image.paste(img, (col * width, row * height))

    # 保存拼接后的图像
    grid_image.save(args.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, default="source1.png")
    parser.add_argument("--image_path", type=str, default='data/style_images_aligned/')
    # parser.add_argument("--layer", type=int, default=11,
    #                     help='Transformer layer from which to extract the feature, between 0-11')
    parser.add_argument("--save_path", type=str, default='outputs/pca/')
    # parser.add_argument("--mode", type=str, default='k')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    args.image_path += args.image
    args.save_path += args.image
    visualize(args)
import os.path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import argparse
from PIL import Image
from torchvision.models import VGG16_Weights
from decimal import Decimal, ROUND_HALF_UP

# 转换为 tensor 并标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 转 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载预训练 VGG16
vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# 切换为评估模式（关闭 dropout/batchnorm）
vgg.eval()
# 只保留 features 部分（即卷积层部分，不包含最后的全连接层）
vgg_features = vgg.features
avg_pool = vgg.avgpool


def get_features(image_path: str):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        feature = vgg_features(img_tensor)
        feature = avg_pool(feature)
        feature = torch.flatten(feature, 1)
    return feature


def get_paired_image_paths(source_dir, target_dir):
    # 获取 source 和 target 文件名，排序确保顺序对应
    source_files = sorted([f for f in os.listdir(source_dir)])
    target_files = sorted([f for f in os.listdir(target_dir)])

    pairs = []
    for s, t in zip(source_files, target_files):
        s_path = os.path.join(source_dir, s)
        t_path = os.path.join(target_dir, t)
        pairs.append((s_path, t_path))
    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='style2')
    args = parser.parse_args()

    ground_source = os.path.join('images/ground_truth', f'source{args.style[-1]}.png')
    ground_target = os.path.join('images/ground_truth', f'target{args.style[-1]}.png')

    source_feature = get_features(ground_source)
    target_feature = get_features(ground_target)
    source_feature /= source_feature.norm(dim=1)
    target_feature /= target_feature.norm(dim=1)
    ground_d = target_feature - source_feature
    ground_d /= ground_d.norm(dim=1)
    inputs_folder = ['DoesFS', 'DiFa', 'jojoGAN', 'MTG', 'oneshotCLIP', 'jojoGAN_pair', 'MTG_pair']
    for input_folder in inputs_folder:
        source_dir = 'images/test_inputs'
        target_dir = os.path.join('images', input_folder, args.style)
        losses = []
        total = 0.0
        pairs = get_paired_image_paths(source_dir, target_dir)
        for s, t in pairs:
            source = get_features(s)
            target = get_features(t)
            source /= source.norm(dim=1)
            target /= target.norm(dim=1)
            d = target - source
            d /= d.norm(dim=1)
            losses.append(F.cosine_similarity(ground_d, d).item())
            total += losses[-1]
            losses[-1] = float(Decimal(str(losses[-1])).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        print(losses)
        print(input_folder + ' average loss: ', Decimal(str(total / len(pairs))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))






import lpips
import torch
import os
import argparse
from decimal import Decimal, ROUND_HALF_UP

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
    loss = lpips.LPIPS(net='vgg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default='style2')
    args = parser.parse_args()
    inputs_folder = ['DoesFS', 'DiFa', 'jojoGAN', 'MTG', 'oneshotCLIP', 'jojoGAN_pair', 'MTG_pair']
    for input_folder in inputs_folder:
        source_dir = 'images/test_inputs'
        target_dir = os.path.join('images', input_folder, args.style)

        pairs = get_paired_image_paths(source_dir, target_dir)
        losses = []
        total = 0.0
        print(pairs)
        for source, target in pairs:
            s = lpips.im2tensor(lpips.load_image(source))
            t = lpips.im2tensor(lpips.load_image(target))

            current_loss = loss(s, t).item()
            total += current_loss
            current_loss = float(Decimal(str(current_loss)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            losses.append(current_loss)
        # print('losses: ', losses)
        print(input_folder + ' average loss: ', Decimal(str(total / len(pairs))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))





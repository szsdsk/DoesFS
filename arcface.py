import argparse
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from decimal import Decimal, ROUND_HALF_UP

# 初始化 ArcFace 模型
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # GPU: ['CUDAExecutionProvider']
app.prepare(ctx_id=0, det_size=(224, 224))


def get_features(image_path: str):
    img = cv2.imread(image_path)
    face = app.get(img)[0]
    feature = face.normed_embedding
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
    parser.add_argument('--input_folder', type=str, default='DoesFS')
    args = parser.parse_args()
    source_dir = 'images/test_inputs'
    loss = []
    for i in range(1, 8):
        target_dir = os.path.join('images', args.input_folder, f'target{i}')

        ground_source = os.path.join('images/ground_truth', f'source{i}.png')
        ground_target = os.path.join('images/ground_truth', f'target{i}.png')

        source_feature = get_features(ground_source)
        target_feature = get_features(ground_target)
        ground_d = target_feature - source_feature
        ground_d /= np.linalg.norm(ground_d)
        print(ground_source, ground_target)

        total = 0.0
        pairs = get_paired_image_paths(source_dir, target_dir)
        for s, t in pairs:
            source = get_features(s)
            target = get_features(t)
            d = target - source
            d /= np.linalg.norm(d)
            cosine = np.dot(d, ground_d) / (np.linalg.norm(d) * np.linalg.norm(ground_d))
            total += cosine
        loss.append(float(Decimal(str(total / len(pairs))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)))
    print(loss)

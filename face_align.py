import cv2 as cv
import dlib
import numpy as np
from PIL import Image
import scipy
import argparse
import torchvision
from torchvision import transforms
import os


# https://gitee.com/zengxy2020/csdn_dlib_face_alignment/blob/master/ffhq_dataset/face_alignment.py
def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    faces = detector(img, 1)
    assert len(faces) > 0, 'face not detected, try another face image'

    shape = predictor(img, faces[0])

    parts = list(shape.parts())
    res = []
    for part in parts:
        res.append([part.x, part.y])
    lm = np.array(res)
    return lm, faces[0]


def align_face(filepath, output_size=1024, enable_padding=True):
    predictor = dlib.shape_predictor("checkpoints/shape_predictor_68_face_landmarks.dat")
    lm, rect = get_landmark(filepath, predictor)
    l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()

    img = cv.imread(filepath)
    cv.rectangle(img, (l, t), (r, b), color=(0, 0, 255))

    # 标出 68 个点
    for index, pt in enumerate(lm):
        pt_pos = tuple(pt)
        cv.circle(img, pt_pos, 2, (0, 0, 255), 2)

    # 颚点 = 0–16 右眉点 = 17–21 左眉点 = 22–26 鼻点 = 27–35
    # 右眼点 = 36–41 左眼点 = 42–47 口角 = 48–59 嘴唇分数 = 60–67
    lm_chin = lm[0: 17]
    lm_eyebrow_left = lm[17:22]
    lm_eyebrow_right = lm[22: 27]
    lm_nose = lm[27: 31]
    lm_nostrils = lm[31: 36]
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary point.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x) # normalize
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    # 与 x 垂直的向量
    y = np.flipud(x) * [-1, 1]
    # 到眼睛下方一点
    c = eye_avg + eye_to_mouth * 0.1
    # 举行的边界
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = Image.open(filepath).convert('RGB')
    transform_size = output_size

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))

    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')[:, :, :3]
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))

        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Return aligned image.
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/test_inputs/012.png')
    parser.add_argument('--output', type=str, default='./outputs/aligned')
    args = parser.parse_args()

    transform = transforms.Compose([transforms.Resize((1024, 1024)),
                                    transforms.ToTensor()])
    face_aligned = align_face(args.path)
    face_aligned = transform(face_aligned)
    os.makedirs(args.output, exist_ok=True)
    torchvision.utils.save_image(face_aligned, os.path.join(args.output, os.path.basename(args.path)),
                                 normalize=True, value_range=(0, 1))

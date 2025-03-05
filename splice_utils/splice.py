import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from splice_utils.extractor import VitExtractor


class Splice:

    def __init__(self, device='cuda'):
        self.device = device
        self.extractor = VitExtractor(model_name='dino_vitb8', device=device)
        self.denorm = transforms.Normalize((-1., -1., -1.), (2., 2., 2.))
        self.global_transform = transforms.Compose([
            # global_resize_transform
            transforms.Resize(224, max_size=480),
            # imagenet_norm
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.get_embedding = {
            'k': self.extractor.get_keys_from_input,
            'q': self.extractor.get_queries_from_input,
            'v': self.extractor.get_values_from_input,
            'f': self.extractor.get_features_from_input
        }
        self.get_ssim = {
            'k': self.extractor.get_keys_self_sim_from_input,
            'q': self.extractor.get_queries_self_sim_from_input,
            'v': self.extractor.get_values_self_sim_from_input,
            'f': self.extractor.get_features_self_sim_from_input
        }

    # 提取图片最后一层的keys的自相似性矩阵
    def ssim(self, x):
        sims = []
        for img in x:
            img = self.denorm(img)
            img = self.global_transform(img)
            # 加一个batch_size
            self_sim = self.extractor.get_keys_self_sim_from_input(img.unsqueeze(0), layer_num=11)
            sims.append(self_sim)
        sims = torch.cat(sims, 0)
        return sims

    # 用余弦相似度mse求loss
    def calculate_global_ssim_loss(self, outputs, inputs, mode='k', layer_num=11):
        assert mode in ['k', 'q', 'v', 'f']
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.denorm(a)
            b = self.denorm(b)
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_self_sim = self.get_ssim[mode](a.unsqueeze(0), layer_num=layer_num)
            self_sim = self.get_ssim[mode](b.unsqueeze(0), layer_num=layer_num)
            loss += F.mse_loss(self_sim, target_self_sim)
        return loss

    # 直接用特征值计算mse loss
    def calculate_sim_loss(self, outputs, inputs, mode='k', layer_num=11):
        assert mode in ['k', 'q', 'v', 'f']
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.denorm(a)
            b = self.denorm(b)
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target = self.get_embedding[mode](a.unsqueeze(0), layer_num=layer_num)
                # head_num, patch_num, dim
                h, t, d = target.shape
                target_ = target.transpose(0, 1).reshape(t, h*d)
            predict = self.get_embedding[mode](b.unsqueeze(0), layer_num=layer_num)
            h, t, d = predict.shape
            predict_ = predict.transpose(0, 1).reshape(t, h*d)
            loss += F.mse_loss(predict_, target_)
        return loss

    # 获得最后一层 cls
    def cls(self, x):
        clss = []
        for a in x:
            a = self.denorm(a)
            a = self.global_transform(a).unsqueeze(0)
            token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            clss.append(token)
        clss = torch.cat(clss, dim=0)
        return clss

    # 直接用最后一层cls的特征值计算mse loss
    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.denorm(a)
            b = self.denorm(b)
            a = self.global_transform(a).unsqueeze(0)
            b = self.global_transform(b).unsqueeze(0)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.denorm(a)
            b = self.denorm(b)
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss

    def calculate_features(self, images, mode='f', layers: list = None):
        if layers is None:
            layers = [3, 5, 7, 9, 11]
        feats = []
        for img in images:
            fs = []
            img = self.denorm(img)
            img = self.global_transform(img)
            for layer_num in layers:
                f = self.get_embedding[mode](img.unsqueeze(0), layer_num=layer_num)
                h, t, d = f.shape
                # 去掉cls
                # f [1, token_num-1, feature_num]
                f = f.transpose(0, 1).reshape(t, h*d)[1:, :].unsqueeze(0)
                fs.append(f)
            # fs [1, (token_num-1) * len(layers), feature_num]
            fs = torch.cat(fs, dim=1)
            feats.append(fs)
        # feats [images_num, (token_num - 1) * len(layers), feature_num]
        feats = torch.cat(feats, dim=0)

        return feats

    def calculate_self_sim(self, images, mode='f', layers: list = None):
        if layers is None:
            layers = [3, 5, 7, 9, 11]
        self_sims = []
        for img in images:
            ssims = []
            img = self.denorm(img)
            img = self.global_transform(img)
            for layer_num in layers:
                # ssim [1, token_num-1, token_num-1]
                ssim = self.get_ssim[mode](img.unsqueeze(0), layer_num=layer_num)
                ssims.append(ssim)
            # token_num - 1 = 785 - 1 = 784
            # ssims [1, (token_num-1) * len(layers), token_num-1]
            ssims = torch.cat(ssims, dim=1)
            self_sims.append(ssims)
        # self_sims [image_num, (token_num-1) * len(layers), token_num-1]
        self_sims = torch.cat(self_sims, dim=0)
        return self_sims

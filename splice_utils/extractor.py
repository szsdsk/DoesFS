import torch


# 自相似性矩阵
def attn_cosine_sim(x, eps=1e-08):
    # x : [1, 1, token_num - 1, feature_num]
    x = x[0]
    # x : [1, token_num - 1, feature_num]
    norm1 = x.norm(dim=2, keepdim=True)
    # norm: [1, token_num - 1, 1]
    # factor: [1, token_num-1, token_num-1]
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    # sim_matrix: [1, token_num-1, token_num-1]
    return sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        self._init_hooks_data()

    def _init_hooks_data(self):
        # ViT-S/16（小模型）：拥有 12 个 Transformer 块（层）。
        # ViT-B/16（基础模型）：同样拥有 12 个 Transformer 块（层）。
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        # hook 用来获得模型某一层的输出
        for block_idx, block in enumerate(self.model.blocks):
            # 每一块之后的输出
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            # 对注意力输出的结果做dropout之后的输出
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 三个矩阵concat
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            # 获得的是attention的输出
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    # 钩子函数
    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, input, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, input, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    def _get_patch_imd_hook(self):
        def _get_attn_output(model, input, output):
            # (attn_output, attn_weight)
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    # 利用钩子获得数据
    def get_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    # 获得vit的patch_size
    def get_patch_size(self):
        return 8 if '8' in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    # 图片的patch数 + [cls]
    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    # 这里没有 batch_size 维度
    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    # 获得给定图片某一层的某个数据
    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_values_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        values = self.get_values_from_qkv(qkv_features, input_img.shape)
        return values

    def get_queries_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        queries = self.get_queries_from_qkv(qkv_features, input_img.shape)
        return queries

    def get_features_from_input(self, input_img, layer_num):
        features = self.get_feature_from_input(input_img)[layer_num]
        return features

    # 求的是每个patch间的相似度，且去掉[CLS] token
    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        # head_num, patch_num, dim
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, 1:, :])
        return ssim_map

    def get_values_self_sim_from_input(self, input_img, layer_num):
        values = self.get_values_from_input(input_img, layer_num=layer_num)
        # head_num, token_num, dim
        h, t, d = values.shape
        concatenated_values = values.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, 1:, :])
        return ssim_map

    def get_queries_self_sim_from_input(self, input_img, layer_num):
        queries = self.get_queries_from_input(input_img, layer_num=layer_num)
        # head_num, token_num, dim
        h, t, d = queries.shape
        concatenated_values = queries.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, 1:, :])
        return ssim_map

    def get_features_self_sim_from_input(self, input_img, layer_num):
        features = self.get_features_from_input(input_img, layer_num=layer_num)
        # head_num, token_num, dim
        h, t, d = features.shape
        concatenated_values = features.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, 1:, :])
        return ssim_map


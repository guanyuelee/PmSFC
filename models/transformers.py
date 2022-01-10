import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from models.diff_aug import DiffAugment


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ConcatMLP2(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, hidden_dim=512, n_layers=3, layers=[], activate=torch.relu,
                 std=0.05):
        super(ConcatMLP2, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        if len(layers) == 0:
            self.layers = []
            for i in range(n_layers - 1):
                self.layers.append(hidden_dim)
            self.layers.append(output_dim)
        else:
            self.layers = layers

        self.std = std
        self.net1 = []
        in_ch = input_dim
        for i in range(len(self.layers)):
            self.net1.append(nn.Linear(in_ch, self.layers[i], bias=True))
            in_ch = self.layers[i]
        self.net1 = nn.ModuleList(self.net1)
        self.act = activate

        self.scalar = nn.Parameter(data=torch.ones(size=(1, 1)), requires_grad=True)

        self.initialize()

    def initialize(self):
        for item in self.parameters():
            torch.nn.init.normal_(item, std=self.std)

    def forward(self, z):
        B, d = z.shape

        x = z.view(-1, d)
        for index, l in enumerate(self.net1):
            x = l(x)
            x = self.act(x)

        return x


class ConcatMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, hidden_dim=512, embed_dim=384, n_layers=3, layers=[]):
        super(ConcatMLP, self).__init__()
        self.output_dim = output_dim
        if len(layers) == 0:
            self.layers = []
            for i in range(n_layers - 1):
                self.layers.append(hidden_dim)
            self.layers.append(embed_dim)
        else:
            self.layers = layers

        self.net1 = []
        in_ch = input_dim
        for i in range(len(self.layers)):
            self.net1.append(nn.Linear(in_ch, self.layers[i]))
            in_ch = self.layers[i]
        self.net1 = nn.ModuleList(self.net1)

        self.net2 = []
        in_ch = input_dim
        for i in range(len(self.layers)):
            self.net2.append(nn.Linear(in_ch, self.layers[i]))
            in_ch = self.layers[i]
        self.net2 = nn.ModuleList(self.net2)

        self.net3 = []
        in_ch = embed_dim
        for i in range(n_layers - 1):
            self.net3.append(nn.Linear(in_ch, hidden_dim))
            in_ch = hidden_dim
        self.net3.append(nn.Linear(in_ch, output_dim))
        self.net3 = nn.ModuleList(self.net3)

        self.initialize()

    def initialize(self):
        for item in self.parameters():
            torch.nn.init.normal_(item, std=0.01)

    def forward(self, z, d_n, m):
        B, d, w, h = z.shape
        assert w == 1 and h == 1

        if isinstance(m, torch.Tensor):
            m_k = m.view(-1, 1)
        elif isinstance(m, float):
            m_k = torch.from_numpy(np.array(m)).cuda()
        elif isinstance(m, np.ndarray):
            m_k = torch.from_numpy(m).cuda()
        else:
            raise NotImplemented()

        x = z.view(-1, d)
        # x = torch.sign(m_k) * x
        for index, l in enumerate(self.net1):
            x = l(x)
            if index != self.n_layers - 1:
                x = gelu(x)

        d_n_2 = d_n.view(-1, d)
        # d_n_2 = torch.sign(m_k) * d_n_2
        for index, l in enumerate(self.net2):
            d_n_2 = l(d_n_2)
            if index != self.n_layers - 1:
                d_n_2 = gelu(d_n_2)

        z_move = x + m_k * d_n_2
        for index, l in enumerate(self.net3):
            z_move = l(z_move)
            if index != self.n_layers - 1:
                z_move = gelu(z_move)

        # z_move = z_move - torch.sum(z_move * d_n.view(-1, d), dim=1, keepdim=True) * d_n.view(-1, d)
        return z_move.view(B, self.output_dim, w, h)



def count_matmul(m, x, y):
    num_mul = x[0].numel() * x[1].size(-1)
    # m.total_ops += torch.DoubleTensor([int(num_mul)])
    m.total_ops += torch.DoubleTensor([int(0)])


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N).cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i + w + 1] = 1
        elif N - i <= w:
            mask[:, :, i, i - w:N] = 1
        else:
            mask[:, :, i, i:i + w + 1] = 1
            mask[:, :, i, i - w:i] = 1
    return mask


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.is_mask = is_mask
        self.remove_mask = False
        self.mask_4 = get_attn_mask(is_mask, 4)
        self.mask_5 = get_attn_mask(is_mask, 5)
        self.mask_6 = get_attn_mask(is_mask, 6)
        self.mask_7 = get_attn_mask(is_mask, 7)
        self.mask_8 = get_attn_mask(is_mask, 8)
        self.mask_10 = get_attn_mask(is_mask, 10)

    def forward(self, x, epoch):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.is_mask:
            if epoch < 60:
                if epoch < 22:
                    mask = self.mask_4
                elif epoch < 32:
                    mask = self.mask_6
                elif epoch < 42:
                    mask = self.mask_8
                else:
                    mask = self.mask_10
                attn = attn.masked_fill(mask.to(attn.get_device()) == 0, -1e9)
            else:
                pass

        attn_raw = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_raw)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, is_mask=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            is_mask=is_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, epoch):
        x = x + self.drop_path(self.attn(self.norm1(x), epoch))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, input_tokens=512, token_dim=1024, embed_dim=384, depth=5, n_cls=20,
                 n_subspaces=10, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(Transformer, self).__init__()
        self.input_tokens = input_tokens
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.n_cls = n_cls
        self.n_subspaces = n_subspaces
        self.l1 = nn.Linear(self.token_dim, self.embed_dim)
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, input_tokens + 2, embed_dim))
        self.cls_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_embed = [
            self.pos_embed_1
        ]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        is_mask = False
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                is_mask=is_mask)
            for i in range(depth)])

        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        trunc_normal_(self.cls_embed, std=.02)

        self.l2 = nn.Linear(self.embed_dim, 1, bias=False)
        self.l3 = nn.Linear(self.embed_dim, self.n_cls, bias=False)
        self.l4 = nn.Linear(self.embed_dim, 1, bias=False)
        self.l5 = nn.Linear(self.embed_dim, self.n_subspaces, bias=False)

    def set_arch(self, x, cur_stage):
        pass

    def forward(self, fmap, epoch=0):
        # input shape z: [bs, input_tokens, token_dim^0.5, token_dim^0.5]
        bs = fmap.shape[0]
        z = fmap.view(-1, self.token_dim)
        x = self.l1(z).view(-1, self.input_tokens, self.embed_dim)
        x = torch.cat([self.cls_embed.repeat([bs, 1, 1]).to(z.get_device()), x], dim=1)
        x = x + self.pos_embed[0].to(z.get_device())
        B = x.size()

        for index, blk in enumerate(self.blocks):
            x = blk(x, epoch)

        B, N, C = x.size()

        x_cls = x[:, 0, :].view(bs, -1)
        res_x3 = self.l3(x_cls)
        res_x2 = self.l2(x_cls)

        x_res = x[:, 2:].view(bs, -1, self.embed_dim)

        return res_x2, res_x3, x_res

class DecAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.is_mask = is_mask
        self.remove_mask = False
        self.mask_4 = get_attn_mask(is_mask, 4)
        self.mask_5 = get_attn_mask(is_mask, 5)
        self.mask_6 = get_attn_mask(is_mask, 6)
        self.mask_7 = get_attn_mask(is_mask, 7)
        self.mask_8 = get_attn_mask(is_mask, 8)
        self.mask_10 = get_attn_mask(is_mask, 10)

    def forward(self, kv, q, epoch):
        B, N, C = kv.shape
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        B_q, N_q, C_q = q.shape
        q = self.q(q).reshape(B_q, N_q, 1, self.num_heads, C_q // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
        if self.is_mask:
            if epoch < 60:
                if epoch < 22:
                    mask = self.mask_4
                elif epoch < 32:
                    mask = self.mask_6
                elif epoch < 42:
                    mask = self.mask_8
                else:
                    mask = self.mask_10
                attn = attn.masked_fill(mask.to(attn.get_device()) == 0, -1e9)
            else:
                pass

        attn_raw = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_raw)

        x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DecBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm, is_mask=0):
        super().__init__()
        self.norm_x = norm_layer(dim)
        self.pre_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, is_mask=is_mask)
        self.post_attn = DecAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, is_mask=is_mask)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_kv = norm_layer(dim)
        self.norm_q = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_mlp = norm_layer(dim)

    def forward(self, x, kv, epoch):
        x = x + self.drop_path(self.pre_attn(self.norm_x(x), epoch))
        x = x + self.drop_path(self.post_attn(kv=self.norm_kv(kv), q=self.norm_q(x), epoch=epoch))
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class FullTransformer(nn.Module):
    def __init__(self, input_tokens=512, token_dim=1024, embed_dim=384, encoder_depth=5,
                 decoder_depth=5, n_cls=20, num_heads=4, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, task_nums=6):
        super(FullTransformer, self).__init__()
        self.input_tokens = input_tokens
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.task_nums = task_nums
        self.n_cls = n_cls
        self.l1 = nn.Linear(self.token_dim, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_tokens + 1, embed_dim))
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.out_embed = nn.Parameter(torch.zeros(self.task_nums, input_tokens + 1, embed_dim))
        self.embeds = [self.pos_embed, self.out_embed]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]
        is_mask = False

        self.enc_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                is_mask=is_mask)
            for i in range(encoder_depth)])

        self.dec_blocks = nn.ModuleList([
            DecBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                is_mask=is_mask)
            for i in range(decoder_depth)])

        self.l2 = nn.Linear(self.embed_dim, 1, bias=False)
        self.l3 = nn.Linear(self.embed_dim, self.n_cls, bias=False)
        self.l4 = nn.Linear(self.embed_dim, 1, bias=False)

        for i in range(len(self.embeds)):
            trunc_normal_(self.embeds[i], std=1.0)

        trunc_normal_(self.cls_embed, std=.02)

    def set_arch(self, x, cur_stage):
        pass

    def forward(self, fmap, which_task, epoch=0):
        # input shape z: [bs, input_tokens, token_dim^0.5, token_dim^0.5]
        bs = fmap.shape[0]
        z = fmap.view(-1, self.token_dim)
        x = self.l1(z).view(-1, self.input_tokens, self.embed_dim)
        x = torch.cat([self.cls_embed.repeat([bs, 1, 1]).to(z.get_device()), x], dim=1)
        x = x + self.embeds[0].to(z.get_device())
        B = x.size()

        for index, blk in enumerate(self.enc_blocks):
            x = blk(x, epoch)

        y = self.embeds[1][which_task].view(-1, self.input_tokens + 1, self.embed_dim).to(z.get_device())

        for index, blk in enumerate(self.dec_blocks):
            y = blk(x=y, kv=x, epoch=epoch)

        y_cls = y[:, 0, :].view(bs, -1)
        res_m = self.l2(y_cls)
        res_d = self.l3(y_cls)

        B, N, C = y.size()

        y_cls = y[:, 1:, :].reshape((bs * self.input_tokens, -1))
        res_s = self.l4(y_cls).reshape((bs, self.input_tokens, 1, 1))

        return res_m, res_d, res_s


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class Discriminator(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, args, img_size=32, patch_size=None, in_chans=3, num_classes=1, embed_dim=None, depth=7,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = args.df_dim  # num_features for consistency with other models
        depth = args.d_depth
        self.args = args
        patch_size = args.patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        num_patches = (args.img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.args.diff_aug is not "None":
            x = DiffAugment(x, self.args.diff_aug, True)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


class ConcatMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, hidden_dim=512, embed_dim=384, n_layers=3, layers=[]):
        super(ConcatMLP, self).__init__()
        self.output_dim = output_dim
        if len(layers) == 0:
            self.layers = []
            for i in range(n_layers - 1):
                self.layers.append(hidden_dim)
            self.layers.append(embed_dim)
        else:
            self.layers = layers

        self.net1 = []
        in_ch = input_dim
        for i in range(len(self.layers)):
            self.net1.append(nn.Linear(in_ch, self.layers[i]))
            in_ch = self.layers[i]
        self.net1 = nn.ModuleList(self.net1)

        self.net2 = []
        in_ch = input_dim
        for i in range(len(self.layers)):
            self.net2.append(nn.Linear(in_ch, self.layers[i]))
            in_ch = self.layers[i]
        self.net2 = nn.ModuleList(self.net2)

        self.net3 = []
        in_ch = embed_dim
        for i in range(n_layers - 1):
            self.net3.append(nn.Linear(in_ch, hidden_dim))
            in_ch = hidden_dim
        self.net3.append(nn.Linear(in_ch, output_dim))
        self.net3 = nn.ModuleList(self.net3)

        self.initialize()

    def initialize(self):
        for item in self.parameters():
            torch.nn.init.normal_(item, std=0.01)

    def forward(self, z, d_n, m):
        B, d, w, h = z.shape
        assert w == 1 and h == 1

        if isinstance(m, torch.Tensor):
            m_k = m.view(-1, 1)
        elif isinstance(m, float):
            m_k = torch.from_numpy(np.array(m)).cuda()
        elif isinstance(m, np.ndarray):
            m_k = torch.from_numpy(m).cuda()
        else:
            raise NotImplemented()

        x = z.view(-1, d)
        x = torch.sign(m_k) * x
        for index, l in enumerate(self.net1):
            x = l(x)

        d_n_2 = d_n.view(-1, d)
        d_n_2 = torch.sign(m_k) * d_n_2
        for index, l in enumerate(self.net2):
            d_n_2 = l(d_n_2)

        z_move = x + m_k * d_n_2
        for index, l in enumerate(self.net3):
            z_move = l(z_move)

        z_move = z_move - torch.sum(z_move * d_n.view(-1, d), dim=1, keepdim=True) * d_n.view(-1, d)
        return z_move.view(B, self.output_dim, w, h)





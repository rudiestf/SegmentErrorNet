import torch
import torch.nn as nn
from torchviz import make_dot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vit_b_16  # 使用预训练的 Vision Transformer 作为模块


class SegErrConvClassifier(nn.Module):
    def __init__(self, in_c1: int = 3, in_c2: int = 1, n_layer: int = 7, in_size: int = 96, n_class: int = 3):
        super(SegErrConvClassifier, self).__init__()
        self.name = 'SegErrConvClassifier_layer_%d' % n_layer
        plain_num = 4
        self.n_layer = n_layer
        self.branch1 = nn.ModuleList()
        self.branch1.append(
            nn.Sequential(
                nn.Conv2d(in_c1, plain_num, 1),
                nn.BatchNorm2d(plain_num),
                nn.ReLU(),
            )
        )
        self.branch2 = nn.ModuleList()
        self.branch2.append(
            nn.Sequential(
                nn.Conv2d(in_c2, plain_num // 2, 3, 1, 1),
                nn.BatchNorm2d(plain_num // 2),
                nn.ReLU(),
            )
        )
        for i in range(self.n_layer):
            self.branch1.append(
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(plain_num * pow(2, i), plain_num * pow(2, i + 1), 3, 1, 1),
                        nn.BatchNorm2d(plain_num * pow(2, i + 1)),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(plain_num * pow(2, i + 1), plain_num * pow(2, i + 1), 3, 1, 1),
                        nn.BatchNorm2d(plain_num * pow(2, i + 1)),
                        nn.ReLU(),
                    ),
                )
            )
            self.branch2.append(
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(plain_num * pow(2, i) // 2, plain_num * pow(2, i + 1) // 2, 5, 1, 2),
                        nn.BatchNorm2d(plain_num * pow(2, i + 1) // 2),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(plain_num * pow(2, i + 1) // 2, plain_num * pow(2, i + 1) // 2, 5, 1, 2),
                        nn.BatchNorm2d(plain_num * pow(2, i + 1) // 2),
                        nn.ReLU(),
                    ),
                )
            )
        self.down_sample = nn.MaxPool2d(2, 2)
        d = plain_num * pow(2, self.n_layer - 1) * pow(in_size // pow(2, self.n_layer - 1), 2)
        self.head = nn.Sequential(
            nn.Linear(d * 2, n_class * self.n_layer),
            nn.ReLU(),
            nn.Linear(n_class * self.n_layer, n_class),
            nn.Softmax(dim=1),
        )
        self.param_init()

    def param_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)

    def forward(self, x1, x2):
        x1 = self.branch1[0](x1)
        x2 = self.branch2[0](x2)
        for idx in range(1, self.n_layer):
            x1 = self.branch1[idx][0](x1)
            x1 = self.branch1[idx][1](x1) + x1
            x1 = self.down_sample(x1)
            x2 = self.branch2[idx][0](x2)
            x2 = self.branch2[idx][1](x2) + x2
            x2 = self.down_sample(x2)
        x2_1 = x1[:, 0: x1.size(1) // 2, :, :] * x2
        x2_2 = x1[:, x1.size(1) // 2: x1.size(1), :, :] * x2
        x2 = torch.concat((x2_1, x2_2), dim=1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.concat((x1, x2), dim=1)
        x = self.head(x)
        return x


########################################################################################################################
# transformer-classifier

class PatchEmbedding(nn.Module):
    """
    将图像分割成固定大小的块（patch），并将其展平为一维向量。
    """
    def __init__(self, in_c=3, patch_size=16, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, n_patches, n_patches]
        x = x.flatten(2)  # [B, embed_dim, n_patches * n_patches]
        x = x.transpose(1, 2)  # [B, n_patches * n_patches, embed_dim]
        return x


class TransformerBlock(nn.Module):
    """
    单个 Transformer 编码器块。
    """
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.mlp(x))
        return x


class VisionTransformer(nn.Module):
    """
    自定义 Vision Transformer 模型。
    """
    def __init__(self, in_c, seq_len, patch_size=16, embed_dim=768, num_heads=8, depth=2):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_c, patch_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len + 1, embed_dim).normal_(std=0.02))

    def forward(self, x, ct):
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        x = torch.cat((ct, x), dim=1)
        x = x + self.pos_embedding
        for block in self.transformer_blocks:
            x = block(x)
        # x = self.norm(x)
        return x


class SegErrTransformerClassifier(nn.Module):
    def __init__(self, num_classes=3, c_in_1=3, c_in_2=1, in_h=96, in_w=96):
        super(SegErrTransformerClassifier, self).__init__()
        self.name = 'SegErrTransformerClassifier'
        patch_size = 8
        assert 0 == in_h % patch_size and 0 == in_w % patch_size, \
            'the input image size (width and height) should be divisible by 16'
        hidden_dim = 1024
        self.tf = VisionTransformer(c_in_1 + c_in_2, (in_h // patch_size) * (in_w // patch_size),
                                    patch_size=patch_size, embed_dim=hidden_dim, num_heads=8, depth=1)
        # 全连接层用于分类
        self.fc = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # self.initialize_weights()

    def forward(self, x1, x2):
        b = x1.size(0)
        assert b == x2.size(0), 'batch inconsistent between x1 and x2'
        ct = self.class_token.expand(b, -1, -1)
        x = self.tf(torch.cat((x1, x2), dim=1), ct)
        # 全连接层进行分类
        x = self.fc(x[:, 0])
        return x

    def initialize_weights(self):
        """初始化模型参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对全连接层使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # 对 LayerNorm 初始化为 1 和 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # 对卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.TransformerEncoderLayer):
                # 对 Transformer 层的权重和偏置进行初始化
                nn.init.kaiming_normal_(m.self_attn.in_proj_weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.self_attn.in_proj_bias, 0)
                nn.init.kaiming_normal_(m.linear1.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.linear1.bias, 0)
                nn.init.kaiming_normal_(m.linear2.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.linear2.bias, 0)


def visualize_model():
    model = SegErrConvClassifier()
    out = model(torch.rand(size=(1, 3, 96, 96)), torch.rand(1, 1, 96, 96))
    dot = make_dot(out, params=dict(model.named_parameters()))
    dot.render('model_graph', format='png')


if '__main__' == __name__:
    # a1 = torch.randn(2, 3, 96, 96)
    # a2 = torch.randn(2, 1, 96, 96)
    # mm = SegErrConvClassifier(3, 1)
    # bb = mm(a1, a2)
    # print(bb)

    # visualize_model()

    # 创建模型实例
    model = SegErrTransformerClassifier(num_classes=3)
    in_t1 = torch.randn(1, 3, 96, 96)
    in_t2 = torch.randn(1, 1, 96, 96)
    # 前向传播
    output = model(in_t1, in_t2)
    print("Output shape:", output.shape)  # 输出形状应为 (1, 3)，表示 3 个分类结果

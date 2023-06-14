from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer
import timm
import torch.nn.init as init
from torchvision import models

# GCN模型
class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)

        self.relu = nn.ReLU()

    def forward(self, point, edge):
        # point : batch * feature_dim
        # edge  : batch * batch
        graph_data = GCN.process_graph(edge)

        output_1 = self.linear_1(point)
        output_1 = self.relu(torch.mm(graph_data, output_1))

        output_2 = self.linear_2(output_1)
        output_2 = self.relu(torch.mm(graph_data, output_2))


        return output_2

    @staticmethod
    def process_graph(edge):
        # 求度矩阵
        degree_matrix = torch.sum(edge, dim=-1, keepdim=False)  # [N]
        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        degree_matrix = degree_matrix.pow(-0.5)
        degree_matrix[degree_matrix == float("inf")] = 0.
        tmp = torch.mm(degree_matrix, edge)
        result = torch.mm(tmp, degree_matrix)
        return result


# vit模型提取图像特征
# 修改vit配置
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            # remove the original norm
            del self.norm

        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(196 * 768, 4096),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, :]

        #outcome = outcome.view(outcome.size(0), outcome.size(1) * outcome.size(2))

        outcome1 = outcome[:, 0, :]
        outcome2 = outcome[:, 1:, :]
        outcome2 = outcome2.view(outcome2.size(0), outcome2.size(1) * outcome2.size(2))
        outcome2 = self.mlp(outcome2)
        return outcome1, outcome2


def vit_samall_patch16(**kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, **kwargs)

    model.load_state_dict(torch.load('./model/vit_base_p16_224.pth'), strict=False)
    return model

def vit_patch16(**kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, **kwargs)

    model.load_state_dict(torch.load('./model/vit_base_p16_224.pth'), strict=False)
    return model
class VCGDH(nn.Module):
    def __init__(self, hash_bit, class_num, pretrained=True):
        super(VCGDH, self).__init__()
        model_alexnet = models.alexnet(pretrained=False)
        pre = torch.load("./model/alexnet.pth")
        model_alexnet.load_state_dict(pre)
        self.featureslocal = model_alexnet.features

        self.newfeatureslocal = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True)
        )
        model_vit = vit_samall_patch16()
        self.featureglobal = model_vit
        self.gcn = GCN(4096*2, 2048, 512)

        self.hash_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, hash_bit)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hash_bit, class_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        flocal = self.featureslocal(x.clone())
        flocal = flocal.view(flocal.size(0), 256*6*6)
        flocal = self.newfeatureslocal(flocal)
        cls_token, fglobal = self.featureglobal(x)

        f_fusion = torch.cat((fglobal, flocal), dim=1)
        # fglobal = fglobal.view(fglobal.size(0), 3072)
        # vit和cnn融合
        f_cos = torch.cosine_similarity(cls_token.unsqueeze(1), cls_token.unsqueeze(0), dim=-1)
        S = torch.zeros(f_cos.shape).cuda()
        S[(torch.arange(f_cos.size(0)).unsqueeze(1), torch.topk(f_cos, 10).indices)] = f_cos[(torch.arange(f_cos.size(0)).unsqueeze(1), torch.topk(f_cos, 10).indices)]
        feature_guide = self.gcn(f_fusion, S)
        real_hash_code = self.hash_layer(feature_guide)
        cla = self.classifier(real_hash_code)
        return real_hash_code, cla


class VCGDH_G(nn.Module):
    def __init__(self, hash_bit, class_num, pretrained=True):
        super(VCGDH_G, self).__init__()
        model_alexnet = models.alexnet(pretrained=False)
        pre = torch.load("./model/alexnet.pth")
        model_alexnet.load_state_dict(pre)
        self.featureslocal = model_alexnet.features

        self.newfeatureslocal = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True)
        )
        model_vit = vit_patch16()
        self.featureglobal = model_vit


        self.hash_layer = nn.Sequential(
            nn.Linear(4096*2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, hash_bit)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hash_bit, class_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        flocal = self.featureslocal(x.clone())
        flocal = flocal.view(flocal.size(0), 256*6*6)
        flocal = self.newfeatureslocal(flocal)
        cls_token, fglobal = self.featureglobal(x)

        f_fusion = torch.cat((fglobal, flocal), dim=1)

        real_hash_code = self.hash_layer(f_fusion)
        cla = self.classifier(real_hash_code)
        return real_hash_code, cla

class VCGDH_V(nn.Module):
    def __init__(self, hash_bit, class_num, pretrained=True):
        super(VCGDH_V, self).__init__()
        model_alexnet = models.alexnet(pretrained=False)
        pre = torch.load("./model/alexnet.pth")
        model_alexnet.load_state_dict(pre)
        self.featureslocal = model_alexnet.features

        self.newfeatureslocal = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True)
        )

        self.gcn = GCN(4096, 1024, 256)
        self.hash_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, hash_bit)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hash_bit, class_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        flocal = self.featureslocal(x.clone())
        flocal = flocal.view(flocal.size(0), 256*6*6)
        flocal = self.newfeatureslocal(flocal)

        f_cos = torch.cosine_similarity(flocal.unsqueeze(1), flocal.unsqueeze(0), dim=-1)
        S = torch.zeros(f_cos.shape).cuda()
        S[(torch.arange(f_cos.size(0)).unsqueeze(1), torch.topk(f_cos, 10).indices)] = f_cos[(torch.arange(f_cos.size(0)).unsqueeze(1), torch.topk(f_cos, 10).indices)]
        feature_guide = self.gcn(flocal, S)
        real_hash_code = self.hash_layer(feature_guide)
        cla = self.classifier(real_hash_code)
        return real_hash_code, cla

class VCGDH_C(nn.Module):
    def __init__(self, hash_bit, class_num, pretrained=True):
        super(VCGDH_C, self).__init__()

        model_vit = vit_patch16()
        self.featureglobal = model_vit
        self.gcn = GCN(4096, 1024, 256)
        self.hash_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, hash_bit)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hash_bit, class_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        cls_token, fglobal = self.featureglobal(x)

        f_cos = torch.cosine_similarity(cls_token.unsqueeze(1), cls_token.unsqueeze(0), dim=-1)
        S = torch.zeros(f_cos.shape).cuda()
        S[(torch.arange(f_cos.size(0)).unsqueeze(1), torch.topk(f_cos, 10).indices)] = f_cos[(torch.arange(f_cos.size(0)).unsqueeze(1), torch.topk(f_cos, 10).indices)]
        feature_guide = self.gcn(fglobal, S)
        real_hash_code = self.hash_layer(feature_guide)
        cla = self.classifier(real_hash_code)
        return real_hash_code, cla

class DSH(nn.Module):
    def __init__(self, hash_bit):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, hash_bit)
        )

        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class AlexnetFc(nn.Module):
    def __init__(self, hash_bit):
        super(AlexnetFc, self).__init__()

        model_alexnet = models.alexnet(pretrained=False)
        pre = torch.load("./model/alexnet.pth")
        model_alexnet.load_state_dict(pre)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x


class ResnetFC(nn.Module):
    def __init__(self, hash_bit):
        super(ResnetFC, self).__init__()
        model_resnet = models.resnet50(pretrained=False)
        pre = torch.load("./model/resnet50.pth")
        model_resnet.load_state_dict(pre)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2,
                                            self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y

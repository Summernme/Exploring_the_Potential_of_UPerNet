import torch
import torch.nn as nn
from models.ulsam_attention import ULSAM
from models.modules import conv3x3_bn_relu

class scene_head(nn.Module):
    def __init__(self, fpn_dim, nr_scene_class):
        super(scene_head, self).__init__()
        self.nr_scene_class = nr_scene_class

        self.conv3x3_bn_relu = conv3x3_bn_relu(fpn_dim, fpn_dim, 1)
        self.ulsam_s = ULSAM(nin=fpn_dim, num_splits=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clsf = nn.Conv2d(fpn_dim, self.nr_scene_class, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv3x3_bn_relu(x)
        # scene_att = self.ulsam_s(x)
        out = self.pool(x)
        out = self.clsf(out)

        return out

class object_head(nn.Module):
    def __init__(self, fpn_dim, nr_object_class):
        super(object_head, self).__init__()
        self.nr_object_class = nr_object_class
        self.conv3x3_bn_relu = conv3x3_bn_relu(fpn_dim, fpn_dim, 1)
        self.ulsam_o = ULSAM(nin=fpn_dim, num_splits=4)
        self.clsf = nn.Conv2d(fpn_dim, self.nr_object_class, kernel_size=1, bias=True)

    def forward(self, x, return_attention_maps=False):
        x = self.conv3x3_bn_relu(x)
        object_att = self.ulsam_o(x)
        out = self.clsf(x*object_att)

        if return_attention_maps:
            return x
        return out


class part_head(nn.Module):
    def __init__(self, fpn_dim, nr_part_class):
        super(part_head, self).__init__()
        self.nr_part_class = nr_part_class
        self.conv3x3_bn_relu = conv3x3_bn_relu(fpn_dim, fpn_dim, 1)
        self.ulsam_p = ULSAM(nin=fpn_dim, num_splits=4)
        self.clsf = nn.Conv2d(fpn_dim, self.nr_part_class, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv3x3_bn_relu(x)
        part_att = self.ulsam_p(x)
        out = self.clsf(x*part_att)

        return out

class material_head(nn.Module):
    def __init__(self, fpn_dim, nr_material_class):
        super(material_head, self).__init__()
        self.nr_material_class = nr_material_class

        self.conv3x3_bn_relu = conv3x3_bn_relu(fpn_dim, fpn_dim, 1)
        self.ulsam_m = ULSAM(nin=fpn_dim, num_splits=4)
        self.clsf = nn.Conv2d(fpn_dim, self.nr_material_class, kernel_size=1, bias=True)

    def forward(self, x, return_attention_maps=False):
        x = self.conv3x3_bn_relu(x)
        material_att = self.ulsam_m(x)
        out = self.clsf(x*material_att)

        if return_attention_maps:
            return x
        return out

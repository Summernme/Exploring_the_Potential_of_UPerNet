import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet, resnext
from lib.nn import SynchronizedBatchNorm2d, PrRoIPool2D
from broden_dataset_utils.joint_dataset import broden_dataset

from models.head import scene_head, object_head, part_head, material_head
from models.modules import *

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    @staticmethod
    def pixel_acc(pred, label, ignore_index=-1):
        """Calculates the pixel accuracy for a prediction.

        Args:
            pred (torch.Tensor): The tensor with predicted labels.
            label (torch.Tensor): The tensor with ground truth labels.
            ignore_index (int, optional): The label index to ignore. Defaults to -1.

        Returns:
            acc (torch.Tensor): The pixel accuracy.
        """
        _, preds = torch.max(pred, dim=1)
        valid = (label != ignore_index).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    @staticmethod
    def part_pixel_acc(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        """Calculates the pixel accuracy for a specific part.

        Args:
            pred_part (torch.Tensor): The tensor with predicted labels for the part.
            gt_seg_part (torch.Tensor): The tensor with ground truth labels for the part.
            gt_seg_object (torch.Tensor): The tensor with ground truth labels for the object.
            object_label (int): The label of the object.
            valid (torch.Tensor): The tensor indicating the validity of the part.

        Returns:
            acc_sum, pixel_sum (torch.Tensor, torch.Tensor): The sum of correct pixels and total pixels.
        """
        mask_object = (gt_seg_object == object_label)
        _, pred = torch.max(pred_part, dim=1)
        acc_sum = mask_object * (pred == gt_seg_part)
        acc_sum = torch.sum(acc_sum.view(acc_sum.size(0), -1), dim=1)
        acc_sum = torch.sum(acc_sum * valid)
        pixel_sum = torch.sum(mask_object.view(mask_object.size(0), -1), dim=1)
        pixel_sum = torch.sum(pixel_sum * valid)
        return acc_sum, pixel_sum 

    @staticmethod
    def part_loss(pred_part, gt_seg_part, gt_seg_object, object_label, valid):
        """Calculates the loss for a specific part.

        Args:
            pred_part (torch.Tensor): The tensor with predicted labels for the part.
            gt_seg_part (torch.Tensor): The tensor with ground truth labels for the part.
            gt_seg_object (torch.Tensor): The tensor with ground truth labels for the object.
            object_label (int): The label of the object.
            valid (torch.Tensor): The tensor indicating the validity of the part.

        Returns:
            loss (torch.Tensor): The loss value.
        """
        mask_object = (gt_seg_object == object_label)
        loss = F.nll_loss(pred_part, gt_seg_part * mask_object.long(), reduction='none')
        loss = loss * mask_object.float()
        loss = torch.sum(loss.view(loss.size(0), -1), dim=1)
        nr_pixel = torch.sum(mask_object.view(mask_object.shape[0], -1), dim=1)
        sum_pixel = (nr_pixel * valid).sum()
        loss = (loss * valid.float()).sum() / torch.clamp(sum_pixel, 1).float()
        return loss


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, loss_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit_dict = nn.ModuleDict()
        if loss_scale is None:
            self.loss_scale = {"object": 1, "part": 0.5, "scene": 0.25, "material": 1}
        else:
            self.loss_scale = loss_scale

        # criterion
        self.crit_dict["object"] = nn.NLLLoss(ignore_index=0)  # ignore background 0
        self.crit_dict["material"] = nn.NLLLoss(ignore_index=0)  # ignore background 0
        self.crit_dict["scene"] = nn.NLLLoss(ignore_index=-1)  # ignore unlabelled -1

    def forward(self, feed_dict, *, seg_size=None):
        if seg_size is None:    # Training mode
            # Define output_switch based on feed_dict's source_idx
            # Output_switch controls the output classes (object, part, scene, material)
            if feed_dict['source_idx'] == 0:
                output_switch = {"object": True, "part": True, "scene": True, "material": False}
            elif feed_dict['source_idx'] == 1:
                output_switch = {"object": False, "part": False, "scene": False, "material": True}
            else:
                raise ValueError
            # Obtain the predictions using the decoder and feature maps from the encoder
            pred = self.decoder(
                self.encoder(feed_dict['img'], return_feature_maps=True),
                output_switch=output_switch
            )

            # Calculate losses using the provided ground truth and predictions
            loss_dict = {}
            if pred['object'] is not None:  # object
                loss_dict['object'] = self.crit_dict['object'](pred['object'], feed_dict['seg_object'])
            if pred['part'] is not None:  # part
                part_loss = 0
                for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                    part_loss += self.part_loss(
                        pred['part'][idx_part], feed_dict['seg_part'],
                        feed_dict['seg_object'], object_label, feed_dict['valid_part'][:, idx_part])

                loss_dict['part'] = part_loss
            if pred['scene'] is not None:  # scene
                loss_dict['scene'] = self.crit_dict['scene'](pred['scene'], feed_dict['scene_label'])
                if loss_dict['scene'].isnan():
                    loss_dict['scene'].unsqueeze(0).fill_(0)
            if pred['material'] is not None:  # material
                loss_dict['material'] = self.crit_dict['material'](pred['material'], feed_dict['seg_material'])
            loss_dict['total'] = sum([loss_dict[k] * self.loss_scale[k] for k in loss_dict.keys()])

            # Calculate metrics using the provided ground truth and predictions
            metric_dict= {}
            if pred['object'] is not None:
                metric_dict['object'] = self.pixel_acc(
                    pred['object'], feed_dict['seg_object'], ignore_index=0)
            if pred['material'] is not None:
                metric_dict['material'] = self.pixel_acc(
                    pred['material'], feed_dict['seg_material'], ignore_index=0)
            if pred['part'] is not None:
                acc_sum, pixel_sum = 0, 0
                for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                    acc, pixel = self.part_pixel_acc(
                        pred['part'][idx_part], feed_dict['seg_part'], feed_dict['seg_object'],
                        object_label, feed_dict['valid_part'][:, idx_part])
                    acc_sum += acc
                    pixel_sum += pixel
                metric_dict['part'] = acc_sum.float() / (pixel_sum.float() + 1e-10)
            if pred['scene'] is not None:
                metric_dict['scene'] = self.pixel_acc(
                    pred['scene'], feed_dict['scene_label'], ignore_index=-1)

            return {'metric': metric_dict, 'loss': loss_dict}
        
        else: # Inference mode
            output_switch = {"object": True, "part": True, "scene": True, "material": True}
            pred = self.decoder(self.encoder(feed_dict['img'], return_feature_maps=True),
                                output_switch=output_switch, seg_size=seg_size)
            return pred


# def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
#     """
#     Returns a 3x3 convolution with padding to preserve spatial dimensions.
#     Args:
#         in_planes (int): number of input channels
#         out_planes (int): number of output channels
#         stride (int): convolution stride, default is 1
#         has_bias (bool): whether to include a bias term, default is False
#     """
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=has_bias)


# def conv3x3_bn_relu(in_planes, out_planes, stride=1):
#     """
#     Returns a 3x3 convolution with padding, followed by batch normalization and a ReLU activation.
#     Args:
#         in_planes (int): number of input channels
#         out_planes (int): number of output channels
#         stride (int): convolution stride, default is 1
#     """
#     return nn.Sequential(
#             conv3x3(in_planes, out_planes, stride),
#             SynchronizedBatchNorm2d(out_planes),
#             nn.ReLU(inplace=True),
#             )

class ModelBuilder:
    def __init__(self):
        pass

    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, arch='resnet50_dilated8', fc_dim=512, weights=''):
        """
        Args:
            arch (str): architecture of the model
                        ('resnet34', 'resnet34_dilated8', 'resnet34_dilated16',
                         'resnet50', 'resnet101', 'resnext101')
            fc_dim (int): number of fully connected dimensions
            weights (str): path to pretrained weights; use pretrained weights if not provided
        Returns:
            net_encoder (nn.Module): encoder architecture constructed
        """

        pretrained = len(weights) == 0
        if arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34_dilated8':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=8)
        elif arch == 'resnet34_dilated16':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet,
                                        dilate_scale=16)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        else:
            raise Exception('Architecture undefined!')

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    def build_decoder(self, nr_classes,
                      arch='ppm_bilinear_deepsup', fc_dim=512,
                      weights='', use_softmax=False):
        """
        Builds and returns a decoder based on the architecture specified.
        Can also load pretrained weights if provided.
        Args:
            nr_classes (int): number of output classes
            arch (str): architecture of the model (default: 'ppm_bilinear_deepsup')
            fc_dim (int): number of fully connected dimensions (default: 512)
            weights (str): path to pretrained weights (default: '')
            use_softmax (bool): whether to use softmax in the final layer of decoder (default: False)
        Returns:
            net_decoder (nn.Module): decoder architecture constructed
        """
        if arch == 'upernet_lite':
            net_decoder = UPerNet(
                nr_classes=nr_classes,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                nr_classes=nr_classes,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, r=16):
#         super(SEBlock, self).__init__()

#         self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
#         self.excitation = nn.Sequential(
#             nn.Linear(in_channels, in_channels // r),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // r, in_channels),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         x = self.squeeze(x) # Global Average Pooling
#         x = x.view(x.size(0), -1)   # Batch size축은 놔두고 나머지를 일렬로 쭉 펴기
#         x = self.excitation(x)
#         x = x.view(x.size(0), x.size(1), 1, 1)  # 원래대로 복구
#         return x
    
# class SubSpace(nn.Module):
#     def __init__(self, nin: int) -> None:
#         super(SubSpace, self).__init__()
#         self.conv_dws = nn.Conv2d(nin, nin, kernel_size=1, stride=1, padding=0, groups=nin)
#         self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
#         self.relu_dws = nn.ReLU(inplace=False)

#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

#         self.conv_point = nn.Conv2d(nin, 1, kernel_size=1, stride=1, padding=0, groups=1)
#         self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
#         self.relu_point = nn.ReLU(inplace=False)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.conv_dws(x)
#         out = self.bn_dws(out)
#         out = self.relu_dws(out)

#         out = self.maxpool(out)

#         out = self.conv_point(out)
#         out = self.bn_point(out)
#         out = self.relu_point(out)

#         m, n, p, q = out.shape
#         out = self.softmax(out.view(m, n, -1))
#         out = out.view(m, n, p, q)

#         out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

#         out = torch.mul(out, x)

#         out += x

#         return out


# class ULSAM(nn.Module):
#     def __init__(self, nin: int, num_splits: int) -> None:
#         super(ULSAM, self).__init__()
#         self.subspaces = SubSpace(nin)
#         assert nin % num_splits == 0

#         self.nin = nin
#         # self.nout = nout
#         # self.h = h
#         # self.w = w
#         self.num_splits = num_splits

#         self.subspaces = nn.ModuleList(
#             [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # group_size = int(self.nin / self.num_splits)

#         # split at batch dimension
#         sub_feat = torch.chunk(x, self.num_splits, dim=1)

#         out = []
#         for idx, _ in enumerate(self.subspaces):
#             out.append(self.subspaces[idx](sub_feat[idx]))

#         out = torch.cat(out, dim=1)

#         return out


# upernet
class UPerNet(nn.Module):
    def __init__(self, nr_classes, fc_dim=4096,
                use_softmax=False, pool_scales=(1, 2, 3, 6),
                fpn_inplanes=(256,512,1024,2048), fpn_dim=256,
                attn_dim=512, ul_dim=1024):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax
        # self.head_in = nn.Sequential(
        #     nn.Conv2d(ul_dim*4, ul_dim*2, kernel_size=1),
        #     nn.Conv2d(ul_dim*2, ul_dim, kernel_size=1),
        #     nn.Conv2d(ul_dim, ul_dim//2, kernel_size=1)
        # )

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]: # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1): # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        # background included. if ignore in loss, output channel 0 will not be trained.
        self.nr_scene_class, self.nr_object_class, self.nr_part_class, self.nr_material_class = \
            nr_classes['scene'], nr_classes['object'], nr_classes['part'], nr_classes['material']
        
        self.scene_head = scene_head(fpn_dim, self.nr_scene_class)
        self.object_head = object_head(fpn_dim, self.nr_object_class)
        self.part_head = part_head(fpn_dim, self.nr_part_class)
        self.material_head = material_head(fpn_dim, self.nr_material_class)


    def forward(self, conv_out, output_switch=None, seg_size=None):

        output_dict = {k: None for k in output_switch.keys()}

        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        roi = [] # fake rois, just used for pooling
        for i in range(input_size[0]): # batch size
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1)) # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                pool_scale(conv5, roi.detach()),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        if output_switch['scene']: # scene
            output_dict['scene'] = self.scene_head(f)

        if output_switch['object'] or output_switch['part'] or output_switch['material']:
            fpn_feature_list = [f]
            for i in reversed(range(len(conv_out) - 1)):
                conv_x = conv_out[i]
                conv_x = self.fpn_in[i](conv_x) # lateral branch

                f = F.interpolate(
                    f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
                f = conv_x + f

                fpn_feature_list.append(self.fpn_out[i](f))
            fpn_feature_list.reverse() # [P2 - P5]

            output_size = fpn_feature_list[0].size()[2:]
            fusion_list = [fpn_feature_list[0]]
            for i in range(1, len(fpn_feature_list)):
                fusion_list.append(F.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear', align_corners=False))
            fusion_out = torch.cat(fusion_list, 1)
            x = self.conv_fusion(fusion_out)

            # if output_switch['scene']: # scene
            #     x = fusion_out*(self.se_block_s(fusion_out))
            #     output_dict['scene'] = self.scene_head(x)
            if output_switch['object']: # object
                output_dict['object'] = self.object_head(x)
            if output_switch['part']: # part
                object_att = self.object_head(x, return_attention_maps=True)
                output_dict['part'] = self.part_head(torch.concat((x, object_att), dim=1))
            if output_switch['material']: # material
                object_att = self.object_head(x, return_attention_maps=True)
                output_dict['material'] = self.material_head(torch.concat((fpn_feature_list[0], object_att), dim=1))

            if self.use_softmax:  # is True during inference
                # inference scene
                x = output_dict['scene']
                x = x.squeeze(3).squeeze(2)
                x = F.softmax(x, dim=1)
                output_dict['scene'] = x

                # inference object, material
                for k in ['object', 'material']:
                    x = output_dict[k]
                    x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
                    x = F.softmax(x, dim=1)
                    output_dict[k] = x

                # inference part
                x = output_dict['part']
                x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
                part_pred_list, head = [], 0
                for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                    n_part = len(broden_dataset.object_part[object_label])
                    _x = F.interpolate(x[:, head: head + n_part], size=seg_size, mode='bilinear', align_corners=False)
                    _x = F.softmax(_x, dim=1)
                    part_pred_list.append(_x)
                    head += n_part
                output_dict['part'] = part_pred_list

            else:   # Training
                # object, scene, material
                for k in ['object', 'scene', 'material']:
                    if output_dict[k] is None:
                        continue
                    x = output_dict[k]
                    x = F.log_softmax(x, dim=1)
                    if k == "scene":  # for scene
                        x = x.squeeze(3).squeeze(2)
                    output_dict[k] = x
                if output_dict['part'] is not None:
                    part_pred_list, head = [], 0
                    for idx_part, object_label in enumerate(broden_dataset.object_with_part):
                        n_part = len(broden_dataset.object_part[object_label])
                        x = output_dict['part'][:, head: head + n_part]
                        x = F.log_softmax(x, dim=1)
                        part_pred_list.append(x)
                        head += n_part
                    output_dict['part'] = part_pred_list

            return output_dict

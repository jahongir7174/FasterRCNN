import torch
from torch.jit import annotations
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign


def pad(k, p):
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p), 1, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 1e-3, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.res = Conv(ch, ch, 3)

    def forward(self, x):
        return self.res(x) + x


class CSPBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(out_ch, out_ch)
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2),
                                         Residual(out_ch // 2),
                                         Residual(out_ch // 2))

    def forward(self, x):
        y = torch.cat((self.conv1(x), self.res_m(self.conv2(x))), dim=1)
        return self.conv3(y)


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        p2 = [Conv(filters[0], filters[1], 6, 2, 2),
              Residual(filters[1]),
              Conv(filters[1], filters[2], 3, 2),
              CSPBlock(filters[2], filters[2])]
        p3 = [Conv(filters[2], filters[3], 3, 2),
              CSPBlock(filters[3], filters[3]),
              CSPBlock(filters[3], filters[3])]
        p4 = [Conv(filters[3], filters[4], 3, 2),
              CSPBlock(filters[4], filters[4]),
              CSPBlock(filters[4], filters[4]),
              CSPBlock(filters[4], filters[4])]
        p5 = [Conv(filters[4], filters[5], 3, 2),
              CSPBlock(filters[5], filters[5]),
              SPP(filters[5], filters[5])]

        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p2, p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h0 = CSPBlock(filters[2], filters[2])
        self.h1 = CSPBlock(filters[2], filters[2])
        self.h2 = CSPBlock(filters[2], filters[2])
        self.h3 = Conv(filters[2], filters[2], 3, 2)
        self.h4 = CSPBlock(filters[2], filters[2])
        self.h5 = Conv(filters[2], filters[2], 3, 2)
        self.h6 = CSPBlock(filters[2], filters[2])
        self.h7 = Conv(filters[2], filters[2], 3, 2)
        self.h8 = CSPBlock(filters[2], filters[2])

    def forward(self, x):
        p2, p3, p4, p5 = x

        h0 = self.h0(self.up(p5) + p4)
        h1 = self.h1(self.up(h0) + p3)
        h2 = self.h2(self.up(h1) + p2)
        h4 = self.h4(self.h3(h2) + h1)
        h6 = self.h6(self.h5(h4) + h0)
        h8 = self.h8(self.h7(h6) + p5)
        return h2, h4, h6, h8


class Backbone(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.net = DarkNet(filters)
        self.fpn = DarkFPN(filters)

        self.p2 = Conv(filters[2], filters[2])
        self.p3 = Conv(filters[3], filters[2])
        self.p4 = Conv(filters[4], filters[2])
        self.p5 = Conv(filters[5], filters[2])

    def forward(self, x):
        p2, p3, p4, p5 = self.net(x)

        p2 = self.p2(p2)
        p3 = self.p3(p3)
        p4 = self.p4(p4)
        p5 = self.p5(p5)

        p2, p3, p4, p5 = self.fpn([p2, p3, p4, p5])
        return {'0': p2, '1': p3, '2': p4, '3': p5}


class RPNHead(torch.nn.Module):
    def __init__(self, ch, num_anchor):
        super().__init__()
        self.cls = torch.nn.Conv2d(ch, num_anchor * 1, 1)
        self.reg = torch.nn.Conv2d(ch, num_anchor * 4, 1)

    def forward(self, x):
        cls = []
        reg = []
        for y in x:
            cls.append(self.cls(y))
            reg.append(self.reg(y))
        return cls, reg


class BoxHead(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.fc = torch.nn.Sequential(torch.nn.Linear(ch, 1024),
                                      torch.nn.SiLU(True),
                                      torch.nn.Linear(1024, 1024),
                                      torch.nn.SiLU(True))

    def forward(self, x):
        return self.fc(x.flatten(start_dim=1))


class Predictor(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cls = torch.nn.Linear(1024, num_classes * 1)
        self.reg = torch.nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        return self.cls(x), self.reg(x)


class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(images):
        image_sizes = torch.jit.annotate(annotations.List[annotations.Tuple[int, int]], [])
        for image_size in [image.shape[-2:] for image in images]:
            assert len(image_size) == 2
            image_sizes.append((image_size[0], image_size[1]))
        return ImageList(images, image_sizes)


class FastRCNN(torch.nn.Module):
    def __init__(self, num_classes,
                 # RPN parameters
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_score_thresh=0.001, box_nms_thresh=0.001, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5, box_batch_size_per_image=512,
                 box_positive_fraction=0.25, box_reg_weights=None):
        super().__init__()
        filters = [3, 64, 128, 256, 512, 1024]

        anchor_sizes = ((16, 32, 64), (32, 64, 128), (64, 128, 256), (128, 256, 512))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn_head = RPNHead(filters[2], rpn_anchor_generator.num_anchors_per_location()[0])
        box_head = BoxHead(filters[2] * 7 ** 2)

        self.backbone = Backbone(filters)
        self.transform = Transform()
        self.rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head,
                                         rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                         rpn_batch_size_per_image, rpn_positive_fraction,
                                         rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        self.roi = RoIHeads(MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2),
                            box_head, Predictor(num_classes), box_fg_iou_thresh,
                            box_bg_iou_thresh, box_batch_size_per_image, box_positive_fraction,
                            box_reg_weights, box_score_thresh, box_nms_thresh, box_detections_per_img)

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        images = self.transform(images)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {'0', features}
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi(features, proposals, images.image_sizes, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return self.eager_outputs(losses, detections)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                delattr(m, 'norm')
                m.forward = m.fuse_forward
        return self

import torch
from torch.jit import annotations
from torchvision.models.detection import image_list


def pad(k, p):
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

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
        self.relu = torch.nn.SiLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 1),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv(out_ch, out_ch)
        residual_m = [Residual(out_ch // 2, add) for _ in range(n)]
        self.res_m = torch.nn.Sequential(*residual_m)

    def forward(self, x):
        return self.conv3(torch.cat((self.conv1(x), self.res_m(self.conv2(x))), dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y = self.res_m(x)
        z = self.res_m(y)
        return self.conv2(torch.cat([x, y, z, self.res_m(z)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, filters, num_dep):
        super().__init__()
        p1 = [Conv(filters[0], filters[1], 6, 2, 2)]
        p2 = [Conv(filters[1], filters[2], 3, 2),
              CSP(filters[2], filters[2], num_dep[0])]
        p3 = [Conv(filters[2], filters[3], 3, 2),
              CSP(filters[3], filters[3], num_dep[1])]
        p4 = [Conv(filters[3], filters[4], 3, 2),
              CSP(filters[4], filters[4], num_dep[2])]
        p5 = [Conv(filters[4], filters[5], 3, 2),
              CSP(filters[5], filters[5], num_dep[0]),
              SPP(filters[5], filters[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p2, p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, filters, num_dep):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h11 = Conv(filters[5], filters[4], 1, 1)
        self.h12 = CSP(2 * filters[4], filters[4], num_dep[0], False)
        self.h13 = Conv(filters[4], filters[3], 1, 1)
        self.h14 = CSP(2 * filters[3], filters[3], num_dep[0], False)
        self.h15 = Conv(filters[3], filters[2], 1, 1)
        self.h16 = CSP(2 * filters[2], filters[2], num_dep[0], False)
        self.h17 = Conv(filters[2], filters[2], 3, 2)
        self.h18 = CSP(2 * filters[2], filters[3], num_dep[0], False)
        self.h19 = Conv(filters[3], filters[3], 3, 2)
        self.h20 = CSP(2 * filters[3], filters[4], num_dep[0], False)
        self.h21 = Conv(filters[4], filters[4], 3, 2)
        self.h22 = CSP(2 * filters[4], filters[5], num_dep[0], False)

    def forward(self, x):
        p2, p3, p4, p5 = x

        h11 = self.h11(p5)
        h12 = self.h12(torch.cat([self.up(h11), p4], 1))

        h13 = self.h13(h12)
        h14 = self.h14(torch.cat([self.up(h13), p3], 1))

        h15 = self.h15(h14)
        h16 = self.h16(torch.cat([self.up(h15), p2], 1))

        h17 = self.h17(h16)
        h18 = self.h18(torch.cat([h17, h15], 1))

        h19 = self.h19(h18)
        h20 = self.h20(torch.cat([h19, h13], 1))

        h21 = self.h21(h20)
        h22 = self.h22(torch.cat([h21, h11], 1))
        return h16, h18, h20, h22


class Backbone(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        self.p2 = Conv(width[2], width[2])
        self.p3 = Conv(width[3], width[2])
        self.p4 = Conv(width[4], width[2])
        self.p5 = Conv(width[5], width[2])
        self.p6 = torch.nn.MaxPool2d(1, 2)

    def forward(self, x):
        p2, p3, p4, p5 = self.net(x)
        p2, p3, p4, p5 = self.fpn([p2, p3, p4, p5])
        p2 = self.p2(p2)
        p3 = self.p3(p3)
        p4 = self.p4(p4)
        p5 = self.p5(p5)
        p6 = self.p6(p5)
        return {'0': p2, '1': p3, '2': p4, '3': p5, '4': p6}


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

        self.fc = torch.nn.Sequential(torch.nn.Linear(ch, ch),
                                      torch.nn.SiLU(),
                                      torch.nn.Linear(ch, ch),
                                      torch.nn.SiLU())

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


class Predictor(torch.nn.Module):
    def __init__(self, ch, num_classes):
        super().__init__()
        self.cls = torch.nn.Linear(ch, num_classes * 1)
        self.reg = torch.nn.Linear(ch, num_classes * 4)

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
            image_sizes.append((image_size[0], image_size[1]))
        return image_list.ImageList(images, image_sizes)


class FasterRCNN(torch.nn.Module):
    def __init__(self, width, depth, num_classes,
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

        from torchvision.ops import MultiScaleRoIAlign
        from torchvision.models.detection.roi_heads import RoIHeads
        from torchvision.models.detection.rpn import RegionProposalNetwork
        from torchvision.models.detection.anchor_utils import AnchorGenerator

        anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3', '4'], 7, 2)
        rpn_head = RPNHead(width[2], rpn_anchor_generator.num_anchors_per_location()[0])
        box_head = BoxHead(width[2])
        cls_head = Predictor(width[2], num_classes)

        self.backbone = Backbone(width, depth)
        self.transform = Transform()
        self.rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head,
                                         rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                         rpn_batch_size_per_image, rpn_positive_fraction,
                                         rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        self.roi = RoIHeads(roi_pool, box_head, cls_head, box_fg_iou_thresh,
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
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def faster_r_cnn_n(num_classes: int = 80):
    depth = [1, 2, 3]
    width = [3, 16, 32, 64, 128, 256]
    return FasterRCNN(width, depth, num_classes)


def faster_r_cnn_s(num_classes: int = 80):
    depth = [1, 2, 3]
    width = [3, 32, 64, 128, 256, 512]
    return FasterRCNN(width, depth, num_classes)


def faster_r_cnn_m(num_classes: int = 80):
    depth = [2, 4, 6]
    width = [3, 48, 96, 192, 384, 768]
    return FasterRCNN(width, depth, num_classes)


def faster_r_cnn_l(num_classes: int = 80):
    depth = [3, 6, 9]
    width = [3, 64, 128, 256, 512, 1024]
    return FasterRCNN(width, depth, num_classes)


def faster_r_cnn_x(num_classes: int = 80):
    depth = [4, 8, 12]
    width = [3, 80, 160, 320, 640, 1280]
    return FasterRCNN(width, depth, num_classes)

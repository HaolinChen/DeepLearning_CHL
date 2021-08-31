import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d, MaxPool2d
from torch import nn
from ..nn.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small, Block, hswish

from .ssd import SSD
from .predictor import Predictor
from .config import cnn_ssd_config as config


def SeperableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding, bias=False),
        # BatchNorm2d(in_channels),
        # nn.ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


class cnn_Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_size, out_size):
        super(cnn_Block, self).__init__()
        self.maxpool = MaxPool2d(2, 2)
        self.conv = Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn = BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.maxpool(x)
        out = self.relu(self.bn(self.conv(out)))

        return out


def create_cnn_ssd(num_classes, is_test=False):
    base_net = nn.Sequential(
        Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 128*160
        BatchNorm2d(8),
        nn.ReLU(),
        # ------------------------
        MaxPool2d(2, 2),  # 64*80
        Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(16),
        nn.ReLU(),
        # ------------------------
        MaxPool2d(2, 2),  # 32*40
        Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(16),
        nn.ReLU(),
        # ------------------------
        MaxPool2d(2, 2),  # 16*20
        Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        BatchNorm2d(32),
        nn.ReLU(),
    )

    source_layer_indexes = [15]

    extras = ModuleList([
        cnn_Block(32, 64),  # 8*10
        cnn_Block(64, 128),  # 4*5
        cnn_Block(128, 128),  # 2*2
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=32, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1),
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=32, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_cnn_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor

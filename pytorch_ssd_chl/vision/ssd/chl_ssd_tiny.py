import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d, MaxPool2d
from torch import nn
from ..nn.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small, Block, hswish

from .ssd import SSD
from .predictor import Predictor
from .config import chl_ssd_tiny_config as config


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

class aideck_Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_size, out_size):
        super(aideck_Block, self).__init__()
        self.maxpool = MaxPool2d(2, 2)
        self.conv1 = Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_size)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_size)
        self.relu2 = nn.ReLU()
        self.conv3 = SeperableConv2d(out_size, out_size)
        self.bn3 = BatchNorm2d(out_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.maxpool(x)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))

        return out


def create_chl_ssd_tiny(num_classes, multi=0.75, is_test=False):
    
    base_net = nn.Sequential(
        Conv2d(1, int(32 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(int(32 * multi)),
        nn.ReLU(),
        Conv2d(int(32 * multi), int(32 * multi), kernel_size=1, stride=1, bias=False),
        BatchNorm2d(int(32 * multi)),
        nn.ReLU(),
        SeperableConv2d(int(32 * multi), int(32 * multi)),
        BatchNorm2d(int(32 * multi)),
        nn.ReLU(),
        # ------------------------
        MaxPool2d(2, 2),  # 120*120
        Conv2d(int(32 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        Conv2d(int(16 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        SeperableConv2d(int(16 * multi), int(16 * multi)),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        # ------------------------
        MaxPool2d(2, 2),  # 60*60
        Conv2d(int(16 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        Conv2d(int(16 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        SeperableConv2d(int(16 * multi), int(16 * multi)),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        # ------------------------
        MaxPool2d(2, 2),  # 30*30
        Conv2d(int(16 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        Conv2d(int(16 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        SeperableConv2d(int(16 * multi), int(16 * multi)),
        BatchNorm2d(int(16 * multi)),
        nn.ReLU(),
        # ------------------------
        # MaxPool2d(2, 2),  # 18*18
        # Conv2d(int(16 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        # BatchNorm2d(int(16 * multi)),
        # nn.ReLU(),
        # Conv2d(int(16 * multi), int(16 * multi), kernel_size=3, stride=1, padding=1, bias=False),
        # BatchNorm2d(int(16 * multi)),
        # nn.ReLU(),
        # SeperableConv2d(int(16 * multi), int(16 * multi)),
        # BatchNorm2d(int(16 * multi)),
        # nn.ReLU(),
    )

    source_layer_indexes = [39]

    extras = ModuleList([
        # aideck_Block(32*multi,16*multi),
        aideck_Block(int(16 * multi), int(16 * multi)),  # 15*15
        aideck_Block(int(16 * multi), int(16 * multi)),  # 7*7
        aideck_Block(int(16 * multi), int(16 * multi)),  # 3*3
        aideck_Block(int(16 * multi), int(16 * multi)),  # 1*1
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=int(16 * multi), out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=4 * 4, kernel_size=3, padding=1),
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=int(16 * multi), out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=int(16 * multi), out_channels=4 * num_classes, kernel_size=3, padding=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def create_chl_ssd_tiny_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor

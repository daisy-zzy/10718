"""Register builders returning one binary logit per issue in LABELS."""
import torch
from torch import nn

from . import LABELS

MODEL_REGISTRY = {}


def register_model(name):
    def decorate(builder):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model: {name}")
        MODEL_REGISTRY[name] = builder
        return builder
    return decorate


def build_model(spec):
    name = spec["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**spec.get("kwargs", {}))


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True))


class IssueClassifier(nn.Module):
    def __init__(self, backbone, features, dropout):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({label: nn.Linear(features, 1) for label in LABELS})

    def forward(self, images):
        features = self.dropout(self.pool(self.backbone(images)).flatten(1))
        return torch.cat([self.heads[label](features) for label in LABELS], dim=1)


@register_model("simple_cnn")
def simple_cnn(width=32, dropout=0.2):
    if width < 8 or width % 8:
        raise ValueError("width must be a positive multiple of 8")
    backbone = nn.Sequential(conv_block(3, width, 2), conv_block(width, width * 2, 2),
                             conv_block(width * 2, width * 4, 2),
                             conv_block(width * 4, width * 4, 2))
    return IssueClassifier(backbone, width * 4, dropout)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.main = nn.Sequential(conv_block(in_channels, out_channels, stride),
                                  nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, out_channels))
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels else
                     nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                   nn.GroupNorm(8, out_channels)))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


@register_model("small_resnet")
def small_resnet(width=32, blocks=(2, 2, 2, 2), dropout=0.2):
    """ResNet-18 style residual stages, reduced width and a 3x3 stem; from scratch."""
    if width < 8 or width % 8 or len(blocks) != 4 or any(type(n) is not int or n < 1 for n in blocks):
        raise ValueError("Use width divisible by 8 and four positive integer stage depths")
    layers = [conv_block(3, width, 2)]
    current = width
    for stage, depth in enumerate(blocks):
        channels = width * 2 ** stage
        for block in range(depth):
            stride = 2 if stage > 0 and block == 0 else 1
            layers.append(BasicBlock(current, channels, stride))
            current = channels
    return IssueClassifier(nn.Sequential(*layers), current, dropout)


class ResNet18Block(nn.Module):
    """Standard two-convolution residual block with BatchNorm."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels else
                     nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                         nn.BatchNorm2d(out_channels)))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


@register_model("resnet18")
def resnet18(dropout=0.2):
    """Full-width ResNet-18 backbone and seven binary heads, from scratch.

    Local construction and checkpoint reload never download weights or require
    torchvision. Existing image preprocessing remains unchanged.
    """
    return _full_resnet((2, 2, 2, 2), dropout)


@register_model("resnet34")
def resnet34(dropout=0.2):
    """Full-width ResNet-34: 3/4/6/3 basic blocks and seven binary heads."""
    return _full_resnet((3, 4, 6, 3), dropout)


def _full_resnet(blocks, dropout):
    """Shared standard ResNet stem, residual stages and initialization."""
    layers = [
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1)]
    current = 64
    for stage, channels in enumerate((64, 128, 256, 512)):
        for block in range(blocks[stage]):
            stride = 2 if stage > 0 and block == 0 else 1
            layers.append(ResNet18Block(current, channels, stride))
            current = channels
    model = IssueClassifier(nn.Sequential(*layers), 512, dropout)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model

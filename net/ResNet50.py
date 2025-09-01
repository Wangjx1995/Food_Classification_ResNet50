from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50 as tv_resnet50, ResNet50_Weights

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        # 1x1 reduce
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        # 3x3
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = norm_layer(width)
        # 1x1 expand
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element list")

        self.groups = groups
        self.base_width = width_per_group

        # Stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages (conv2_x .. conv5_x)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool):
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Optionally zero-initialize last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # ===== Utilities =====
    def freeze_stages(self, upto: int = 2):
        """Freeze parameters up to a stage index (inclusive).
        Stages mapping: 0=stem, 1=layer1, 2=layer2, 3=layer3, 4=layer4
        Example: upto=2 freezes stem, layer1, layer2.
        """
        assert 0 <= upto <= 4
        # Stem
        if upto >= 0:
            for m in [self.conv1, self.bn1]:
                for p in m.parameters():
                    p.requires_grad = False
        # Stages
        if upto >= 1:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if upto >= 2:
            for p in self.layer2.parameters():
                p.requires_grad = False
        if upto >= 3:
            for p in self.layer3.parameters():
                p.requires_grad = False
        if upto >= 4:
            for p in self.layer4.parameters():
                p.requires_grad = False

    def replace_classifier(self, num_classes: int, bias: bool = True):
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, num_classes, bias=bias)


def resnet50(
    num_classes: int = 1000,
    *,
    pretrained: bool = False,
    pretrained_source: str = "torchvision",
    freeze_upto: int | None = None,
    **kwargs
) -> ResNet:
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

    if pretrained:
        if pretrained_source == "torchvision":

            tv = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            state = tv.state_dict()

            if num_classes != 1000:
                state.pop("fc.weight", None)
                state.pop("fc.bias", None)
                model.load_state_dict(state, strict=False)
            else:
                model.load_state_dict(state, strict=True)

    if freeze_upto is not None:
        model.freeze_stages(upto=freeze_upto)

    return model

# ===== Quick smoke test =====
if __name__ == "__main__":
    model = resnet50(num_classes=10, zero_init_residual=True)
    model.freeze_stages(upto=2)  # freeze stem + layer1 + layer2
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print("Output shape:", logits.shape)  # (2, 10)

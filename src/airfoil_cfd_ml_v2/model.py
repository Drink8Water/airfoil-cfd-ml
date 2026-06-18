from __future__ import annotations

import torch
import torch.nn as nn


def block_unet(
    in_channels: int,
    out_channels: int,
    name: str,
    size: int = 4,
    pad: int = 1,
    transposed: bool = False,
    use_batch_norm: bool = True,
    activation: bool = True,
    relu: bool = True,
    dropout: float = 0.0,
):
    block = nn.Sequential()

    if not transposed:
        block.add_module(
            f"{name}_conv",
            nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=2, padding=pad, bias=True),
        )
    else:
        block.add_module(f"{name}_upsample", nn.Upsample(scale_factor=2, mode="bilinear"))
        block.add_module(
            f"{name}_tconv",
            nn.Conv2d(in_channels, out_channels, kernel_size=(size - 1), stride=1, padding=pad, bias=True),
        )

    if use_batch_norm:
        block.add_module(f"{name}_bn", nn.BatchNorm2d(out_channels))
    if dropout > 0.0:
        block.add_module(f"{name}_dropout", nn.Dropout2d(dropout, inplace=True))

    if activation:
        if relu:
            block.add_module(f"{name}_relu", nn.ReLU(inplace=True))
        else:
            block.add_module(f"{name}_leakyrelu", nn.LeakyReLU(0.2, inplace=True))

    return block


class DfpNet(nn.Module):
    def __init__(self, channel_exponent: int = 6, dropout: float = 0.0):
        super().__init__()
        channels = int(2**channel_exponent + 0.5)

        self.layer1 = block_unet(3, channels * 1, "enc_layer1", transposed=False, use_batch_norm=True, relu=False, dropout=dropout)
        self.layer2 = block_unet(channels, channels * 2, "enc_layer2", transposed=False, use_batch_norm=True, relu=False, dropout=dropout)
        self.layer3 = block_unet(channels * 2, channels * 2, "enc_layer3", transposed=False, use_batch_norm=True, relu=False, dropout=dropout)
        self.layer4 = block_unet(channels * 2, channels * 4, "enc_layer4", transposed=False, use_batch_norm=True, relu=False, dropout=dropout)
        self.layer5 = block_unet(channels * 4, channels * 8, "enc_layer5", transposed=False, use_batch_norm=True, relu=False, dropout=dropout)
        self.layer6 = block_unet(channels * 8, channels * 8, "enc_layer6", transposed=False, use_batch_norm=True, relu=False, dropout=dropout, size=2, pad=0)
        self.layer7 = block_unet(channels * 8, channels * 8, "enc_layer7", transposed=False, use_batch_norm=True, relu=False, dropout=dropout, size=2, pad=0)

        self.dlayer7 = block_unet(channels * 8, channels * 8, "dec_layer7", transposed=True, use_batch_norm=True, relu=True, dropout=dropout, size=2, pad=0)
        self.dlayer6 = block_unet(channels * 16, channels * 8, "dec_layer6", transposed=True, use_batch_norm=True, relu=True, dropout=dropout, size=2, pad=0)
        self.dlayer5 = block_unet(channels * 16, channels * 4, "dec_layer5", transposed=True, use_batch_norm=True, relu=True, dropout=dropout)
        self.dlayer4 = block_unet(channels * 8, channels * 2, "dec_layer4", transposed=True, use_batch_norm=True, relu=True, dropout=dropout)
        self.dlayer3 = block_unet(channels * 4, channels * 2, "dec_layer3", transposed=True, use_batch_norm=True, relu=True, dropout=dropout)
        self.dlayer2 = block_unet(channels * 4, channels, "dec_layer2", transposed=True, use_batch_norm=True, relu=True, dropout=dropout)
        self.dlayer1 = block_unet(channels * 2, 3, "dec_layer1", transposed=True, use_batch_norm=False, activation=False, dropout=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)

        dout6 = self.dlayer7(out7)
        dout6 = self.dlayer6(torch.cat([dout6, out6], 1))
        dout5 = self.dlayer5(torch.cat([dout6, out5], 1))
        dout4 = self.dlayer4(torch.cat([dout5, out4], 1))
        dout3 = self.dlayer3(torch.cat([dout4, out3], 1))
        dout2 = self.dlayer2(torch.cat([dout3, out2], 1))
        dout1 = self.dlayer1(torch.cat([dout2, out1], 1))
        return dout1


def weights_init(module: nn.Module):
    classname = module.__class__.__name__
    if "Conv" in classname and hasattr(module, "weight"):
        module.weight.data.normal_(0.0, 0.02)
    elif "BatchNorm" in classname and hasattr(module, "weight"):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

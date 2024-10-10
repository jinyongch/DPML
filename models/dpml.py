import torch
from thop import profile
from torch import nn
from torchsummary import summary

try:
    from osnet import OSBlock, OSNet
except ImportError:
    from Networks.DPML.osnet import OSBlock, OSNet


def build_osnet(model_type="1_0", pretrained=True):
    if model_type == "1_0":
        channels = [64, 256, 384, 512]
    elif model_type == "0_5":
        channels = [32, 128, 192, 256]

    osnet = OSNet(
        num_classes=1000,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=channels,
        loss="softmax",
    )

    if pretrained:
        model_path = f"pretrained/osnet_x{model_type}_imagenet.pth"
        state_dict = torch.load(model_path)
        osnet.load_state_dict(state_dict)

    osnet = nn.Sequential(*list(osnet.children())[:-3])

    return osnet


class DPML(nn.Module):
    def __init__(self):
        super().__init__()

        self.master = build_osnet(model_type="1_0")
        self.proxy = build_osnet(model_type="0_5")

        self.align = nn.Conv2d(256, 512, 3, 1, 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, 0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, 0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, 0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, 0, bias=True),
            nn.ReLU(),
        )

    def frezze(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_master = self.master(x)
        output_master = self.decoder(x_master)

        x_proxy = self.proxy(x)
        x_proxy = self.align(x_proxy)
        output_proxy = self.decoder(x_proxy)

        return output_master, output_proxy, x_master, x_proxy

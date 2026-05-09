# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import timm

from domainbed.lib import misc
from domainbed.lib import wide_resnet
from domainbed.lib import big_transfer
from domainbed.lib import vision_transformer
from domainbed.lib import mlp_mixer

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['backbone'] == 'resnet10':
            self.network = torchvision.models.resnet10(pretrained=True)
            self.n_outputs = 512
            self.disable_bn = True
        if hparams['backbone'] == 'resnet18':
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet50':
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet18-BN':
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
            self.disable_bn = False
        elif hparams['backbone'] == 'resnet50-BN':
            from domainbed import Res 
            self.network = Res.__dict__['resnet50'](pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = False
        elif hparams['backbone'] == 'resnet50-GN':
            self.network = timm.create_model('resnet50_gn', pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = False
        elif hparams['backbone'] == 'resnet101':
            self.network = timm.create_model('resnet101', pretrained=True)
            self.n_outputs = 2048
            self.disable_bn = False

        if self.disable_bn:
            self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        if hparams.get("use_image_net_pretrained", False):
            self.classifier = copy.deepcopy(self.network.fc)
        
        del self.network.fc
        self.network.fc = Identity()
        # save memory
        if self.disable_bn:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.disable_bn:
            self.freeze_bn()
 
    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)
    
class MonaiDenseNet(nn.Module):
    def __init__(self, input_shape, hparams):
        super().__init__()
        import monai.networks.nets as monai_nets

        self.network = monai_nets.DenseNet121(
            spatial_dims=3,
            in_channels=input_shape[0],
            out_channels=1,   # replaced below
            pretrained=False,
        )
        # Remove classification head → return pooled feature vector
        self.network.class_layers.out = Identity()
        self.n_outputs = 1024
        self.dropout = nn.Dropout(hparams.get("densenet_dropout", 0.0))
    
    def forward(self, x):
        return self.dropout(self.network(x))

class MonaiViT(nn.Module):
    _CONFIGS = {
        "monai-ViT": {"hidden_size": 768, "num_heads": 12, "patch_size": 16, "mlp_dim": 3072},
        "monai-ViT-B16": {"hidden_size": 768, "num_heads": 12, "patch_size": 16, "mlp_dim": 3072},
        "monai-ViT-S16": {"hidden_size": 384, "num_heads": 6, "patch_size": 16, "mlp_dim": 1536},
        "monai-ViT-T16": {"hidden_size": 192, "num_heads": 3, "patch_size": 16, "mlp_dim": 768},
        "monai-ViT-M16": {"hidden_size": 96, "num_heads": 2, "patch_size": 16, "mlp_dim": 384},
    }
    
    def __init__(self, input_shape, hparams):
        super().__init__()
        import monai.networks.nets as monai_nets
        config = self._CONFIGS.get(hparams['backbone'], None)
        if config is None:
            raise ValueError(
                f"Unknown MONAI ViT backbone '{hparams['backbone']}'. "
                f"Choose from: {list(self._CONFIGS.keys())}"
            )
        print(f"Using MONAI ViT with patch size {config['patch_size']}x{config['patch_size']}x{config['patch_size']}")
        self.network = monai_nets.ViT(
            spatial_dims=len(input_shape) - 1,
            in_channels=input_shape[0],
            img_size=input_shape[1:],
            num_classes=1,   # replaced below
            classification=True,
            **config,
        )
        # Remove classification head → return pooled feature vector
        self.network.classification_head = Identity()
        self.n_outputs = config['hidden_size']
        self.dropout = nn.Dropout(hparams.get("densenet_dropout", 0.0))
    
    def forward(self, x):
        return self.dropout(self.network(x)[0])

class MonaiResNet(nn.Module):
    """3-D ResNet featurizer via MONAI for volumetric (C, D, H, W) inputs.

    Supported backbones (set hparams['backbone']):
      'monai-resnet10'  -  n_outputs = 512
      'monai-resnet18'  -  n_outputs = 512
      'monai-resnet50'  -  n_outputs = 2048
    """

    _CONFIGS = {
        "monai-resnet10": ("resnet10", 512),
        "monai-resnet18": ("resnet18", 512),
        "monai-resnet50": ("resnet50", 2048),
    }

    def __init__(self, input_shape, hparams):
        super().__init__()
        import monai.networks.nets as monai_nets

        backbone = hparams["backbone"]
        if backbone not in self._CONFIGS:
            raise ValueError(
                f"Unknown MONAI backbone '{backbone}'. "
                f"Choose from: {list(self._CONFIGS.keys())}"
            )
        fn_name, self.n_outputs = self._CONFIGS[backbone]
        n_input_channels = input_shape[0]

        print(f"Using MONAI ResNet with backbone {fn_name} and {n_input_channels} input channels and {len(input_shape) - 1} spatial dims")

        self.network = getattr(monai_nets, fn_name)(
            spatial_dims=len(input_shape) - 1,
            n_input_channels=n_input_channels,
            num_classes=1,   # replaced below
            pretrained=False,
        )
        # Remove classification head → return pooled feature vector
        self.network.fc = Identity()
        self.dropout = nn.Dropout(hparams.get("resnet_dropout", 0.0))

    def forward(self, x):
        return self.dropout(self.network(x))


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif hparams['backbone'].startswith('monai-resnet'):
        return MonaiResNet(input_shape, hparams)
    elif hparams['backbone'].startswith('monai-densenet'):
        return MonaiDenseNet(input_shape, hparams)
    elif hparams['backbone'].startswith('monai-ViT'):
        return MonaiViT(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and hparams['backbone'] in ['resnet50', 'resnet18', 'resnet50-BN', 'resnet50-GN', 'resnet18-BN', 'resnet101']:
        return ResNet(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'ViT-' in hparams['backbone']:
        return vision_transformer.ViT2(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and hparams['backbone'] in ['B_16', 'B_32', 'L_16', 'L_32']:
        return vision_transformer.ViT(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'dino' in hparams['backbone']:
        return vision_transformer.DINO(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'DeiT' in hparams['backbone']:
        return vision_transformer.DeiT(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'HViT' in hparams['backbone']:
        return vision_transformer.HybridViT(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'Mixer' in hparams['backbone']:
        return mlp_mixer.MLPMixer(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'BiT' in hparams['backbone']:
        return big_transfer.BiT(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

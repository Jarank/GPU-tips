from torchvision.models.alexnet import alexnet
from torchvision.models.convnext import convnext_base, convnext_large, convnext_small, convnext_tiny
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201
from torchvision.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.models.regnet import (
    regnet_x_16gf,
    regnet_x_1_6gf,
    regnet_x_32gf,
    regnet_x_3_2gf,
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_x_8gf,
    regnet_y_128gf,
    regnet_y_16gf,
    regnet_y_1_6gf,
    regnet_y_32gf,
    regnet_y_3_2gf,
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_8gf,
)
from torchvision.models.resnet import (
    resnet101,
    resnet152,
    resnet18,
    resnet34,
    resnet50,
    resnext101_32x8d,
    resnext101_64x4d,
    resnext50_32x4d,
    wide_resnet101_2,
    wide_resnet50_2,
)
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    fcn_resnet101,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)
from torchvision.models.shufflenetv2 import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.swin_transformer import swin_b, swin_s, swin_t
from torchvision.models.vgg import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)
from torchvision.models.vision_transformer import vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import time
MODEL_LIST = {
    "alexnet": [alexnet],
    "convnext": [convnext_base, convnext_large, convnext_small, convnext_tiny],
    "densenet": [densenet121, densenet161, densenet169, densenet201],
    "efficientnet": [
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
        efficientnet_v2_l,
        efficientnet_v2_m,
        efficientnet_v2_s,
    ],
    "googlenet": [googlenet],
    "inception": [inception_v3],
    "mnasnet": [mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3],
    "mobilenetv2": [mobilenet_v2],
    "mobilenetv3": [mobilenet_v3_large, mobilenet_v3_small],
    "regnet": [
        regnet_x_16gf,
        regnet_x_1_6gf,
        regnet_x_32gf,
        regnet_x_3_2gf,
        regnet_x_400mf,
        regnet_x_800mf,
        regnet_x_8gf,
        regnet_y_128gf,
        regnet_y_16gf,
        regnet_y_1_6gf,
        regnet_y_32gf,
        regnet_y_3_2gf,
        regnet_y_400mf,
        regnet_y_800mf,
        regnet_y_8gf,
    ],
    "resnet": [
        resnet101,
        resnet152,
        resnet18,
        resnet34,
        resnet50,
        resnext101_32x8d,
        resnext101_64x4d,
        resnext50_32x4d,
        wide_resnet101_2,
        wide_resnet50_2,
    ],
    "shufflenetv2": [
        shufflenet_v2_x0_5,
        shufflenet_v2_x1_0,
        shufflenet_v2_x1_5,
        shufflenet_v2_x2_0,
    ],
    "squeezenet": [squeezenet1_0, squeezenet1_1],
    "swin_transformer": [swin_b, swin_s, swin_t],
    "vgg": [vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn],
    "vision_transformer": [vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32],
}


def train(precision, device, loader, args):
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_name, models in MODEL_LIST.items():
        for model_funcition in models:
            model = model_funcition(True)
            if args.NUM_GPU > 1:
                model = nn.DataParallel(model, device_ids=range(args.NUM_GPU))
            model = getattr(model, precision)()
            model = model.to(device)
            durations = []
            print(f"Benchmarking Training {precision} precision type {model_name} ")
            for step, img in enumerate(loader):
                img = getattr(img, precision)()
                img = img.to(device)
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img)
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= args.WARM_UP:
                    durations.append((end - start) * 1000)
            print(f"{model_name} model average train time : {sum(durations)/len(durations)}ms")
            benchmark[model_name] = durations
    return benchmark


def inference(precision, device, loader, args):
    benchmark = {}
    with torch.no_grad():
        for model_name, models in MODEL_LIST.items():
            for model_funcition in models:
                model = model_funcition()(True)
                if args.NUM_GPU > 1:
                    model = nn.DataParallel(model, device_ids=range(args.NUM_GPU))
                model = getattr(model, precision)()
                model = model.to(device)
                model.eval()
                durations = []
                print(f"Benchmarking Inference {precision} precision type {model_name} ")
                for step, img in enumerate(rand_loader):
                    img = getattr(img, precision)()
                    torch.cuda.synchronize()
                    img = img.to(device)
                    start = time.time()
                    model(img)
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start) * 1000)
                print(
                    f"{model_name} model average inference time : {sum(durations)/len(durations)}ms"
                )
                benchmark[model_name] = durations
    return benchmark


class RandomDataset(Dataset):
    def __init__(self, length):
        self.len = length

    def __getitem__(self, index):
        return torch.randn(3, 224, 224)

    def __len__(self):
        return self.len

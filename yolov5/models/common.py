from utils.general.coordinate import xyxy2xywh
from utils.general.yaml import yaml_load
import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils import TryExcept

from utils.general.constant import ROOT
from utils.general.checks import is_jupyter
from utils.general.str import colorstr
from utils.general.directories import increment_path

from utils.plot.annotator import Annotator, save_one_box
from utils.plot.color import colors

# kernel, padding, dilation


def autopad(k, p=None, d=1):
    # Pad to 'same' shape outputs
    if d > 1:
        if isinstance(k, int):
            k = d * (k-1) + 1
        else:
            k = [d * (x - 1) + 1 for x in k]

    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = [x // 2 for x in k]

    return p

# Standard Conv Layer


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(
            k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# Standard bottleneck
class Bottleneck(nn.Module):
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# CSP Bottleneck dengan 3 Convolutions Layer
class C3(nn.Module):
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# Spatial Pyramid Pooling (SPP) layer
# referensi: https://arxiv.org/abs/1406.4729
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

        modulse = [nn.MaxPool2d(kernel_size=x, stride=1,
                                padding=x // 2) for x in k]
        self.m = nn.ModuleList(modulse)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# Spatial Pyramid Pooling - Fast (SPPF) layer
# Referensi: Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)

            # Gabung tensor sequential dalam dimensi dan ukuran yang sama
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# Focus
class Focus(nn.Module):
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x):
        # x(b, c, w, h) -> y(b, 4c, w/2, h/2)

        cat1 = x[..., ::2, ::2]
        cat2 = x[..., 1::2, ::2]
        cat3 = x[..., ::2, 1::2]
        cat4 = x[..., 1::2, 1::2]

        # Gabung tensor sequential dalam dimensi dan ukuran yang sama
        return self.conv(torch.cat((cat1, cat2, cat3, cat4), 1))


# Contract width-height into channels,
# i.e. x(1, 64, 80, 80) to x(1, 256, 40, 40)
class Contract(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()

        s = self.gain

        # x(1, 64, 40, 2, 40, 2)
        x = x.view(b, c, h // s, s, w // s, s)

        # x(1, 2, 2, 64, 40, 40)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()

        # x(1, 256, 40, 40)
        return x.view(b, c * s * s, h // s, w // s)


# Expand channels into width-height,
# i.e. x(1, 64, 80, 80) to x(1, 16, 160, 160)
class Expand(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()

        s = self.gain

        # x(1, 2, 2, 16, 80, 80)
        x = x.view(b, s, s, c // s ** 2, h, w)

        # x(1, 16, 80, 2, 80, 2)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()

        # x(1, 16, 160, 160)
        return x.view(b, c // s ** 2, h * s, w * s)


# Concatenate a list of tensors along dimension
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):

    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), data=None, fp16=False, fuse=True):
        from models.experimental import attempt_load

        super().__init__()
        stride = 32  # default stride

        if isinstance(weights, list):
            w = str(weights[0])
        else:
            w = str(weights)

        model = attempt_load(weights if isinstance(
            weights, list) else w, device=device, inplace=True, fuse=fuse)
        stride = max(int(model.stride.max()), 32)  # model stride

        if hasattr(model, 'module'):
            names = model.modules.names
        else:
            names = model.names

        if fp16:
            model.half()
        else:
            model.float()

        # explicitly assign for to(), cpu(), cuda(), half()
        self.model = model

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {
                i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            # human-readable names
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']

        self.__dict__.update(locals())  # assign all variables to self

    # YOLOv5 MultiBackend inference

    def forward(self, im, augment=False, visualize=False):
        b, ch, h, w = im.shape  # batch, channel, height, width

        if self.fp16 and im.dtype != torch.float16:
            # to FP16
            im = im.half()

        # if self.nhwc:
            # torch BCHW to numpy BHWC shape(1,320,192,3)
            # im = im.permute(0, 2, 3, 1)

        # Output Model
        if augment or visualize:
            y = self.model(im, augment=augment, visualize=visualize)
        else:
            y = y = self.model(im)

        if isinstance(y, (list, tuple)):
            if len(y) == 1:
                return self.from_numpy(y[0])
            else:
                return [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        else:
            return x

    # Warmup model by running inference once
    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.fp16:
            dtype = torch.half
        else:
            dtype = torch.float

        # input
        im = torch.empty(*imgsz, dtype=dtype, device=self.device)

        self.forward(im)  # warmup

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            # assign stride, names
            return d['stride'], d['names']

        return None, None


# YOLOv5 detections class for inference results
class Detections:

    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        # device
        d = pred[0].device

        # normalizations
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d)
              for im in ims]

        # list of images as numpy arrays
        self.ims = ims

        # list of tensors pred[0] = (xyxy, conf, cls)
        self.pred = pred

        # class names
        self.names = names

        # image filenames
        self.files = files

        # profiling times
        self.times = times

        # xyxy pixels
        self.xyxy = pred

        # xywh pixels
        self.xywh = [xyxy2xywh(x) for x in pred]

        # xyxy normalized
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]

        # xywh normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]

        # number of images (batch size)
        self.n = len(self.pred)

        # timestamps (ms)
        self.t = tuple(x.t / self.n * 1E3 for x in times)

        # inference BCHW shape
        self.s = tuple(shape)

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            # string
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '

            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    # detections per class
                    n = (pred[:, -1] == c).sum()

                    # add to string
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                s = s.rstrip(', ')

                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    # xyxy, confidence, class
                    for *box, conf, cls in reversed(pred):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / \
                                self.names[int(cls)] / \
                                self.files[i] if save else None

                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})

                        else:  # all others
                            annotator.box_label(
                                box, label if labels else '', color=colors(cls))

                    im = annotator.im
            else:
                s += '(no detections)'

            if isinstance(im, np.ndarray):
                im = Image.fromarray(im.astype(np.uint8))
            else:
                im = im

            if show:
                if is_jupyter():
                    from IPython.display import display
                    display(im)
                else:
                    im.show(self.files[i])

            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    print(
                        f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")

            if render:
                self.ims[i] = np.asarray(im)

        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t

        if crop:
            if save:
                print(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        # show results
        self._run(show=True, labels=labels)

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        # increment save_dir
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)

        # save results
        self._run(save=True, labels=labels, save_dir=save_dir)

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):

        if save:
            save_dir = increment_path(save_dir, exist_ok, mkdir=True)
        else:
            save = None

        # crop results
        return self._run(crop=True, save=save, save_dir=save_dir)

    def render(self, labels=True):
        # render results
        self._run(render=True, labels=labels)
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):

            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]]
                  for x in x.tolist()] for x in getattr(self, k)]  # update

            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])

        return new

    def tolist(self):
        # iterable
        r = range(self.n)
        x = [Detections([self.ims[i]], [self.pred[i]], [
                        self.files[i]], self.times, self.names, self.s) for i in r]
        return x

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        # print results
        return self._run(pprint=True)

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()

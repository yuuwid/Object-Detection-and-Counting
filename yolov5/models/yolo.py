from utils.general.core import make_divisible
from utils.general.str import colorstr
from utils.torch.device import select_device
from utils.torch.model import model_info, fuse_conv_and_bn, initialize_weights
from utils.torch.image import scale_img
from utils.torch.time import time_sync

from utils.plot.visualize import feature_visualization

from utils.general.version import check_version

from models.experimental import *
from models.common import *
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# YOLOv5 Detect Head for Detection models
class Detect(nn.Module):

    # strides computed during build
    stride = None

    # force grid reconstruction
    dynamic = False

    # export mode
    export = False

    # detection layer
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        # number of classes
        self.nc = nc

        # number of outputs per anchor
        self.no = nc + 5

        # number of detection layers
        self.nl = len(anchors)

        # number of anchors
        self.na = len(anchors[0]) // 2

        # init grid
        self.grid = [torch.empty(0) for _ in range(self.nl)]

        # init anchor grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]

        # shape(nl, na, 2)
        tensor = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', tensor)

        # output conv
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

        # use inplace ops (e.g. slice assignment)
        self.inplace = inplace

    def forward(self, x):
        # inference output
        z = []

        for i in range(self.nl):
            # conv
            x[i] = self.m[i](x[i])

            # x(bs, 255, 20, 20) ---> x(bs, 3, 20, 20, 85)
            bs, _, ny, nx = x[i].shape

            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            # inference
            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(
                        nx, ny, i)

                # Detect (boxes only)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]

                y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype

        # grid shape
        shape = 1, self.na, ny, nx, 2

        x = torch.arange(nx, device=d, dtype=t)
        y = torch.arange(ny, device=d, dtype=t)

        if torch_1_10:
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            # torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x)

        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5

        anchor_grid = (self.anchors[i] * self.stride[i])
        anchor_grid = anchor_grid.view((1, self.na, 1, 1, 2)).expand(shape)

        return grid, anchor_grid


# YOLOv5 base model
class BaseModel(nn.Module):

    def forward(self, x, profile=False, visualize=False):
        # single-scale inference, train
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            # if not from previous layer
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    # from earlier layers
                    x = [x if j == -1 else y[j] for j in m.f]

            if profile:
                self._profile_one_layer(m, x, dt)

            # run
            x = m(x)

            # save output
            y.append(x if m.i in self.save else None)

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        return x

    def _profile_one_layer(self, m, x, dt):
        # is final layer, copy input as inplace fix
        c = m == self.model[-1]
        o = 0
        t = time_sync()

        for _ in range(10):
            m(x.copy() if c else x)

        dt.append((time_sync() - t) * 100)

        if m == self.model[0]:
            print(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")

        print(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

        if c:
            print(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # fuse model Conv2d() + BatchNorm2d() layers
    def fuse(self):
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv)) and hasattr(m, 'bn'):
                # update conv
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                # remove batchnorm
                delattr(m, 'bn')
                # update forward
                m.forward = m.forward_fuse

        self.info()

        return self

    # print model information
    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()

        if isinstance(m, (Detect)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))

            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))

        return self

from utils.anchor.auto import check_anchor_order

# YOLOv5 detection model
class DetectionModel(BaseModel):

    # model, input channels, number of classes
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml
            self.yaml_file = Path(cfg).name

            with open(cfg, encoding='ascii', errors='ignore') as f:
                # model dict
                self.yaml = yaml.safe_load(f)

        ### Define model ###

        # input channels
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)

        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            # override yaml value
            self.yaml['nc'] = nc

        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            
            # override yaml value
            self.yaml['anchors'] = round(anchors)  

        # model, savelist
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  

        # default names
        self.names = [str(i) for i in range(self.yaml['nc'])]  
        
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        
        if isinstance(m, (Detect)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            def forward(x): return self.forward(
                x)[0] if isinstance(m) else self.forward(x)

            # forward
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])

            check_anchor_order(m)

            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            # only run once
            self._initialize_biases()  

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        # single-scale inference, train
        return self._forward_once(x, profile, visualize)

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            # de-scale
            p[..., :4] /= scale  
            
            if flips == 2:
                # de-flip ud
                p[..., 1] = img_size[0] - p[..., 1]  
            elif flips == 3:
                # de-flip lr
                p[..., 0] = img_size[1] - p[..., 0]  
        else:
            # de-scale
            x, y = p[..., 0:1] / scale, p[..., 1:2] / scale
            wh = p[..., 2:4] / scale  
            
            if flips == 2:
                # de-flip ud
                y = img_size[0] - y  
            
            elif flips == 3:
                # de-flip lr
                x = img_size[1] - x  
            
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        
        return p

    # initialize biases into Detect(), cf is class frequency
    # https://arxiv.org/abs/1708.02002 section 3.3
    def _initialize_biases(self, cf=None):
        # Detect() module
        m = self.model[-1]  
        
        for mi, s in zip(m.m, m.stride):
            # conv.bias(255) --> conv.bias(3, 85)
            b = mi.bias.view(m.na, -1)  

            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            
            # cls
            if cf is None:
                b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999))
            else:
                b.data[:, 5:5 + m.nc] +=        torch.log(cf / cf.sum())

            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


def parse_model(d, ch):
    # Parse a YOLOv5 model.yaml dictionary
    msg = f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
    print(msg)

    anchors = d['anchors']
    nc = d['nc']
    gd = d['depth_multiple']
    gw = d['width_multiple']
    act = d.get('activation')

    if act:
        # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        Conv.default_act = eval(act)

        print(f"{colorstr('activation:')} {act}")

    if isinstance(anchors, list):
        na = (len(anchors[0]) // 2)
    else:
        ns = anchors

    # number of outputs = anchors * (classes + 5)
    no = na * (nc + 5)

    # layers, savelist, ch out
    layers, save, c2 = [], [], ch[-1]

    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # eval strings
        m = eval(m) if isinstance(m, str) else m

        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                # eval strings
                args[j] = eval(a) if isinstance(a, str) else a

        # depth gain
        n = n_ = max(round(n * gd), 1) if n > 1 else n

        layers = {
            Conv, Bottleneck, SPP, SPPF, MixConv2d, Focus,
            C3, nn.ConvTranspose2d
        }
        if m in layers:
            c1, c2 = ch[f], args[0]

            # if not output
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {C3}:
                # number of repeats
                args.insert(2, n)
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m in {Detect}:
            args.append([ch[x] for x in f])

            if isinstance(args[1], int):
                # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        elif m is Contract:
            c2 = ch[f] * args[0] ** 2

        elif m is Expand:
            c2 = ch[f] // args[0] ** 2

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))
                           ) if n > 1 else m(*args)  # module

        # module type
        t = str(m)[8:-2].replace('__main__.', '')

        # number params
        np = sum(x.numel() for x in m_.parameters())

        # attach index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np

        msg = f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}'
        print(msg)

        # append to savelist
        save.extend(x % i for x in (
            [f] if isinstance(f, int) else f) if x != -1)

        layers.append(m_)

        if i == 0:
            ch = []

        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)

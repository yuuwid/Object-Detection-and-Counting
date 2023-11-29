
from utils.general.coordinate import xyxy2xywh, xywh2xyxy
from utils.general.directories import increment_path
from utils.general.boxes import clip_boxes
import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.general.checks import is_ascii, is_jupyter
from utils.general.constant import FONT
from utils.general.config import CONFIG_DIR
from utils.general.image import scale_image

# Settings
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):

        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'

        # non-latin labels, i.e. asian, arabic, cyrillic
        non_ascii = not is_ascii(example)
        self.pil = pil or non_ascii

        self.im = im

        # line width
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color,
                      thickness=self.lw, lineType=cv2.LINE_AA)

        if label:
            # font thickness
            tf = max(self.lw - 1, 1)

            # text width, height
            w, h = cv2.getTextSize(
                label, 0, fontScale=self.lw / 3, thickness=tf)[0]

            outside = p1[1] - h >= 3

            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled

            cv2.putText(self.im,
                        label, (p1[0], p1[1] -
                                2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(
                1, 2, 0).contiguous().cpu().numpy() * 255

        colors = torch.tensor(colors, device=im_gpu.device,
                              dtype=torch.float32) / 255.0

        # shape(n, 1, 1, 3)
        colors = colors[:, None, None]

        # shape(n, h, w, 1)
        masks = masks.unsqueeze(3)
        # shape(n, h, w, 3)
        masks_color = masks * (colors * alpha)

        # shape(n, h, w, 1)
        inv_alph_masks = (1 - masks * alpha).cumprod(0)

        # mask color summand shape(n, h, w, 3)
        mcs = (masks_color * inv_alph_masks).sum(0) * 2

        # flip channel
        im_gpu = im_gpu.flip(dims=[0])

        # shape(h,w,3)
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255).byte().cpu().numpy()
        self.im[:] = im_mask if retina_masks else scale_image(
            im_gpu.shape, im_mask, self.im.shape)
        if self.pil:
            # convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image
        if anchor == 'bottom':  # start y from font bottom
            # text width, height
            w, h = self.font.getsize(text)
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        if isinstance(im, Image.Image):
            self.im = im
        else:
            self.im = Image.fromarray(im)

        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)

    # boxes
    b = xyxy2xywh(xyxy)

    if square:
        # attempt rectangle to square
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)

    # box wh * gain + pad
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]

    if save:
        # make directory
        file.parent.mkdir(parents=True, exist_ok=True)

        f = str(increment_path(file).with_suffix('.jpg'))

        # cv2.imwrite(f, crop)
        # # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f,
                                              quality=95, subsampling=0)  # save RGB

    return crop

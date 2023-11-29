
import math

import torch.nn.functional as F


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16, 3, 256, 416)
    # Scales img(bs, 3, y, x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img

    h, w = img.shape[2:]

    s = (int(h * ratio), int(w * ratio))  # new size

    # resize - interpolasi
    img = F.interpolate(img, 
                        size=s, 
                        mode='bilinear',
                        align_corners=False)  

    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))

    # value = imagenet mean
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)

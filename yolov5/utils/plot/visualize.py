
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


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            # filename
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"
            f = save_dir / f

            # select batch index 0, block by channels
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)
            n = min(n, channels)  # number of plots

            # 8 rows x n/8 cols
            num_plot = math.ceil(n / 8)
            fig, ax = plt.subplots(num_plot, 8, tight_layout=True)  
            
            ax = ax.ravel()
            
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save


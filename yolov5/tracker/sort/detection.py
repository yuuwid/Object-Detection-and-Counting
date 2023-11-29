import numpy as np


class Detection(object):

    """
    Parameters
    ----------
    tlwh : array
        Top-Left Width-Height
        Bounding Box dengan format (x, y, w, h)
    confidence : float
        Confidence Score Detection
    feature : array
        Feature pendeskripsi suatu objek dalam gambar     
    
    """
    def __init__(self, tlwh, confidence, feature) -> None:
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    # tlbr = Top-Left Bottom-Right
    def to_tlbr(self):
        # Convert Bbox ke format (min x, min y, max x, max y)
        # format baru --> (top-left, bottom-right)
        newBbox = self.tlwh.copy()
        newBbox[2:] += newBbox[:2]
        return newBbox
    
    # xyah = X-Y Aspect_Ratio-Height
    def to_xyah(self):
        # Convert Bbox ke format (center x, center y, aspect ratio, height)
        # Aspect Ratio = Width / Height
        # Center X = X + Width / 2
        # Center Y = Y + Height / 2
        newBbox = self.tlwh.copy()
        newBbox[:2] += newBbox[2:] / 2
        newBbox[2] /= newBbox[3]
        return newBbox
    
    
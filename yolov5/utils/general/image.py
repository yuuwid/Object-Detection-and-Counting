import cv2


def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        # gain  = old / new
        x1 = im1_shape[0] / im0_shape[0]
        x2 = im1_shape[1] / im0_shape[1]
        gain = min(x1, x2)  
        
        # wh padding
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  
    
    else:
        pad = ratio_pad[1]
    
    top, left = int(pad[1]), int(pad[0])  # y, x
    
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    
    masks = masks[top:bottom, left:right]

    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

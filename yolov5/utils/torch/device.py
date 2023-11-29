import os
import torch

def select_device(device="", batch_size=0, newline=True):

    # to string, 'cuda:0' to '0'
    device = str(device).strip().lower()
    device = device.replace('cuda:', '').replace('none', '')

    # Check CPU
    cpu = False
    if device == 'cpu':
        cpu = True

    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        
        assert_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', ''))
        assert_msg = f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

        assert assert_cuda, assert_msg

    if not cpu and torch.cuda.is_available():
        if device:
            device = device.split(',')
        else:
            device = '0'
        
        n = len(device)
        if n > 1 and batch_size > 0:
            assert_cond = batch_size % n == 0
            assert_msg = f'batch-size {batch_size} not multiple of GPU count {n}'

            assert assert_cond, assert_msg

        arg = 'cuda:0'
        
    # revert to CPU
    else: 
        arg = 'cpu'
    
    return torch.device(arg)

import pkg_resources as pkg
from utils.general.str import emojis


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    
    # bool
    result = (current == minimum) if pinned else (current >= minimum)  
    
    # string
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'
    
    if hard:
        # assert min requirements met
        assert result, emojis(s)  
    
    return result

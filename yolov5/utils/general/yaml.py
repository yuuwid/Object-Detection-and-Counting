import yaml
from pathlib import Path



def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        for k, v in data.items():
            if (isinstance(v, Path)):
                data = {k: str(v)}
            else:
                data = {k: v}
            yaml.safe_dump(data, f, sort_keys=False)
        # yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)

import pathlib as pth
import numpy as np
from tqdm import tqdm

import pathlib as pth
from tqdm import tqdm

SWAP = {15: 16, 16: 15}

def swap_labels(root: pth.Path) -> None:
    for p in tqdm(list(root.rglob('*.npy'))):
        label = int(p.stem.rsplit('_', 1)[-1])
        if label not in SWAP:
            continue
        new_name = p.stem.rsplit('_', 1)[0] + f'_{SWAP[label]}.npy'
        p.rename(p.parent / new_name)

if __name__ == '__main__':
    swap_labels(pth.Path('/mnt/DATA_SSD/BRIK/TREE_CLASS/GRAJEWO/processed'))
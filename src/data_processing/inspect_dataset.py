import numpy as np
import pathlib as pth

def main(path: str | pth.Path):
    path = pth.Path(path)

    labels = [int(p.stem.rsplit('_', 1)[-1]) for p in list(path.rglob('*.npy'))]
    labels = np.asarray(labels)
    
    unique = np.unique(labels, return_counts=True)
    for label, count in zip(unique[0], unique[1]):
        print(label, count)

if __name__ == '__main__':
    main('/mnt/DATA_SSD/BRIK/TREE_CLASS/GRAJEWO/processed/train')
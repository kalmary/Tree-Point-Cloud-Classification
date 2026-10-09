import argparse
import pathlib as pth

from tqdm import tqdm

SWAP = {15: 16, 16: 15}

def swap_labels(root: pth.Path) -> None:
    for p in tqdm(list(root.rglob('*.npy'))):
        try:
            label = int(p.stem.rsplit('_', 1)[-1])
        except Exception:
            continue
        
        if label not in SWAP:
            continue
        new_name = p.stem.rsplit('_', 1)[0] + f'_{SWAP[label]}.npy'
        p.rename(p.parent / new_name)

def argparser(argv=None):
    parser = argparse.ArgumentParser(description="Swap configured labels in NumPy filenames.")
    parser.add_argument("--source-path", type=pth.Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = argparser(argv)
    swap_labels(args.source_path)


if __name__ == '__main__':
    main()

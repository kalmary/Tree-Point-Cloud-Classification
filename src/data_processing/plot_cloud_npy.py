import argparse
import pathlib as pth
from typing import Optional

import numpy as np


def plot_cloud(points: np.ndarray, labels: Optional[np.ndarray] = None):
    import pyvista as pv
    
    cloud = pv.PolyData(points)
    
    # Konfiguracja renderowania w przeglądarce
    pv.set_jupyter_backend('trame')  # Działa też w zwykłych skryptach .py
    
    # Tworzenie plottera
    plotter = pv.Plotter()
    
    if labels is not None:
        # Add cluster labels as scalar data
        cloud['cluster'] = labels
        plotter.add_mesh(
            cloud, 
            scalars='cluster',
            cmap='tab20',  # Good colormap for discrete clusters
            point_size=5, 
            render_points_as_spheres=True
        )
    else:
        # Original behavior - single color
        plotter.add_mesh(
            cloud, 
            color='green', 
            point_size=5, 
            render_points_as_spheres=True
        )
    
    # Wyświetlenie - to otworzy przeglądarkę
    plotter.show()
    
    plotter.close()
    plotter.deep_clean()

def argparser(argv=None):
    parser = argparse.ArgumentParser(description="Plot tree point clouds stored as NumPy arrays.")
    parser.add_argument("--input-path", type=pth.Path, required=True)
    parser.add_argument("--label", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = argparser(argv)
    if args.input_path.is_file():
        file_paths = [args.input_path]
    elif args.label is None:
        file_paths = list(args.input_path.glob("*.npy"))
    else:
        file_paths = [
            file
            for file in args.input_path.glob("*.npy")
            if f"_{args.label}.npy" in file.name
        ]

    for file in file_paths:
        pcd = np.load(file)
        plot_cloud(pcd)


if __name__ == '__main__':
    main()

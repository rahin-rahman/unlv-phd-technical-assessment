"""
Schematic Grid Visualization of Strict City-Level Geographic Partitioning Protocol.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.plotting import set_publication_style, save_figure


def plot_spatial_split_schematic(manifest_path="outputs/tables/part2_spatial_split_manifest.csv"):
    set_publication_style()
    df = pd.read_csv(manifest_path)
    
    cities = ['Austin', 'Chicago', 'Kitsap', 'Tyrol-w', 'Vienna']
    n_cities = len(cities)
    n_tiles = 36
    
    grid = np.zeros((n_cities, n_tiles))
    split_map = {'train': 0, 'validation': 1, 'test_city': 2}
    
    for _, row in df.iterrows():
        c_idx = cities.index(row['city'])
        t_id = row['tile_id']
        t_num = int(''.join([c for c in t_id if c.isdigit()]))
        t_idx = t_num - 1
        grid[c_idx, t_idx] = split_map[row['split']]
        
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.matplotlib.colors.ListedColormap(['#2b5c8f', '#e67e22', '#27ae60'])
    
    im = ax.imshow(grid, cmap=cmap, aspect='auto')
    
    ax.set_xticks(range(n_tiles))
    ax.set_xticklabels(range(1, n_tiles + 1), fontsize=9)
    ax.set_yticks(range(n_cities))
    ax.set_yticklabels([
        'Austin  [TRAIN]',
        'Chicago  [TRAIN]',
        'Kitsap  [TRAIN]',
        'Tyrol-w  [VALIDATION]',
        'Vienna  [HELD-OUT TEST]'
    ], fontweight='bold', fontsize=11)
    
    ax.set_xlabel("Tile Index (1 to 36 per city)", fontweight='bold')
    ax.set_ylabel("Geographic Region / City Partition", fontweight='bold')
    ax.set_title("Schematic Diagram: Strict City-Level Geographic Generalization Protocol", fontweight='bold', fontsize=14)
    
    # Horizontal boundary line separating train, val, and test cities
    ax.axhline(y=2.5, color='#e67e22', linestyle='--', linewidth=2, label='Train / Validation City Boundary')
    ax.axhline(y=3.5, color='#27ae60', linestyle='-', linewidth=2.5, label='Validation / Test City Generalization Barrier')
    
    ax.set_xticks(np.arange(-.5, n_tiles, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_cities, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)
    
    train_patch = mpatches.Patch(color='#2b5c8f', label='TRAIN: Austin, Chicago, Kitsap (108 Tiles)')
    val_patch = mpatches.Patch(color='#e67e22', label='VALIDATION: Tyrol-w (36 Tiles)')
    test_patch = mpatches.Patch(color='#27ae60', label='TEST: Vienna (36 Tiles Held-Out)')
    
    ax.legend(handles=[train_patch, val_patch, test_patch], loc='upper right', bbox_to_anchor=(1.0, 1.25), ncol=3, frameon=True)
    
    plt.tight_layout()
    path = save_figure(fig, "fig6_spatial_split_schematic.png", "outputs/figures")
    print(f"Schematic Spatial Split Map saved to: {path}")
    return path


if __name__ == '__main__':
    plot_spatial_split_schematic()

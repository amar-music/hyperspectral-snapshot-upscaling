# Build validation set for training
from preprocessing import preprocessSnapshot
import glob
import os
import numpy as np

SQUARE = True

# Specify paths
path = "data/raw/Snapshot/processed/test/"
ss_names = os.listdir(path)
ss_files = glob.glob("data/raw/Snapshot/processed/test/**/")


# Specify camera calibration files
mtx_path = 'data/raw/calibration/snapshot_matrix.npy'
dist_path = 'data/raw/calibration/snapshot_dist.npy'


if SQUARE == True:
    ss_x_off = (3, 213)
    ss_y_off = (2, 212)

else:
    ss_x_off = (3, 402)
    ss_y_off = (2, 212)


for f in ss_files:
    print(f"Processing file: {ss_names[ss_files.index(f)]}... ({str(ss_files.index(f)+1)}/{str(len(ss_files))})")
    processed_img, wavelengths = preprocessSnapshot(ss_path=f, 
                                                    mtx_path=mtx_path, 
                                                    dist_path=dist_path,
                                                    ss_x_off=ss_x_off,
                                                    ss_y_off=ss_y_off)

    # Save as array
    np.save("data/processed/full_hsi/test/bonus_lr/" + ss_names[ss_files.index(f)] + ".npy", processed_img)

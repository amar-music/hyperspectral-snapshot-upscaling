# Build validation set for training
from preprocessing import preprocessSnapshot
import glob
import numpy as np


# Specify paths
ss_files = glob.glob("data/Snapshot/processed/train/**/")

# Specify camera calibration files
mtx_path = 'calibration/snapshot_matrix.npy'
dist_path = 'calibration/snapshot_dist.npy'



for f in ss_files:
    print("Processing file: " + f + "... (" + str(ss_files.index(f)+1) + "/" + str(len(ss_files)) + ")")
    processed_img, wavelengths = preprocessSnapshot(ss_path=f, 
                                                    mtx_path=mtx_path, 
                                                    dist_path=dist_path)

    # Save as array
    np.save("data/Processed/train/" + f[30:-1] + ".npy", processed_img)

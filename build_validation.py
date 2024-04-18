# Build validation set for training
from preprocessing import preprocessFullHSI
import glob

# Load all hdf5 files in a directory
# hdf5_files = glob.glob('data/FX10/*.hdf5')

# Load 1 hdf5 file as test
hdf5_files = ['data/FX10/spelt1.hdf5']

# Specify camera calibration files
mtx_path = 'data/FX10/2D/hrHSI_matrix.npy'
dist_path = 'data/FX10/2D/hrHSI_dist.npy'

for file in hdf5_files:
    preprocessFullHSI(file, 'data/camera_mtx.npy', 'data/camera_dist.npy', print_info=True)




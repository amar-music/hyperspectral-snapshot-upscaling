# Build validation set for training
from preprocessing import preprocessFullHSI
import glob
import numpy as np

# Load all hdf5 files in a directory
hdf5_files = glob.glob('data/FX10/*.hdf5')

# Load 1 hdf5 file as test
# hdf5_files = ['data/FX10/spelt1_l.hdf5']

# Specify camera calibration files
mtx_path = 'calibration/hrHSI_matrix.npy'
dist_path = 'calibration/hrHSI_dist.npy'

# Snapshot dimensions and wavelengths
ss_shape = (210, 399, 24)
ss_wavelengths = [667, 679, 691, 703, 715, 
                  727, 739, 751, 763, 775, 
                  787, 799, 811, 823, 835, 
                  847, 859, 871, 883, 895, 
                  907, 919, 931, 943]


# Set offsets and rotations
hr_x_off = (106, 1018)   # Full: (0, 1084)
hr_y_off = (189, 720)   # Full: (0, 1015)
rot = -0.2
shear = 0.017


for f in hdf5_files:
    print("Processing file: " + f + "... (" + str(hdf5_files.index(f)+1) + "/" + str(len(hdf5_files)) + ")")
    processed_img, wavelengths = preprocessFullHSI(path_to_hdf5=f, 
                                                   mtx_path=mtx_path, 
                                                   dist_path=dist_path, 
                                                   hr_x_off=hr_x_off, 
                                                   hr_y_off=hr_y_off, 
                                                   rot=rot,
                                                   shear=shear,
                                                   ss_shape=ss_shape, 
                                                   ss_wavelengths=ss_wavelengths)

    # Save as array
    np.save("data/Processed/val/" + f[10:-5] + ".npy", processed_img)


hr_wavelenghts = [666, 679, 690, 704, 715, 
                  726, 739, 750, 764, 775, 
                  786, 800, 811, 822, 835, 
                  846, 860, 871, 882, 896, 
                  907, 918, 932, 943]




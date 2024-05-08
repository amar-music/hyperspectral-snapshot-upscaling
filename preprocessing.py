import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import spectral.io.envi as envi
from scipy import ndimage
import glob
from skimage import transform


# hrHSI preprocessing
def preprocessHrHSI(path_to_hdf5, mtx_path, dist_path, print_info = False):

    # Open the HDF5 file
    with h5py.File(path_to_hdf5, 'r') as f:
        if print_info:

            # List the names of all datasets in the file
            print("Datasets in the file:")

            for name in f.keys():
                print(name)

                # List the attributes of the dataset
                data = f[name]
                print("Attributes of the dataset:")

                for key in data.attrs.keys():
                    print(f"{key}: {data.attrs[key]}")


        # get the dataset from the file (this is how we save the objects downstairs)
        dataset = f['hypercube']
        hcube = f['hypercube'][:]
        
        # Normalize hypercube
        hcube = hcube / np.max(hcube)

        # Flip hypercube to align with snapshot
        hcube = np.flip(hcube, axis=1)
        hcube = np.flip(hcube, axis=2)

        # Rearrange hypercube to be in the shape of (x, y, wavelength)
        hcube = np.moveaxis(hcube, 0, -1)

        # Load wavelengths as integers
        wavelengths = np.array(dataset.attrs['wavelength_nm']).astype(int)
        wavelengths = wavelengths.tolist()


        # Load camera matrix
        mtx = np.load(mtx_path)
        dist = np.load(dist_path)



        # Undistort the hypercube
        h, w = hcube.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv.undistort(hcube, mtx, dist, None, newcameramtx)

 
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]


        return dst, wavelengths


# Show hrHSI image
def previewHrHSI(img, wavelengths, selected_pixel, selected_spectrum):

    # Print array as image
    plt.figure()
    plt.imshow(img[:, :, selected_spectrum])
    plt.plot(selected_pixel[0], selected_pixel[1], 'ro')
    plt.title('Image on Wavelength ' + str(wavelengths[selected_spectrum]) + 'nm')
    plt.show()


    # Print spectral composition of a single pixel
    spectrum = img[selected_pixel[0], selected_pixel[1], :]

    # Plot the spectrum
    plt.figure()
    plt.plot(wavelengths, spectrum)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Spectral Composition of Pixel ' + str(selected_pixel))
    plt.axvline(wavelengths[selected_spectrum], color='r', linestyle='-')
    plt.show()


# Snapshot preprocessing
def preprocessSnapshot(ss_path, mtx_path, dist_path):

    # Open image
    ss_file = envi.open(glob.glob(ss_path + ("*.hdr"))[0], glob.glob(ss_path + "*.raw")[0])

    # Load as numpy array
    ss_img = np.array(ss_file.load())

    # Normalize image
    # hcube = ss_img / np.max(ss_img)
    hcube = ss_img

    # Load wavelengths
    wavelengths = ss_file.metadata['wavelength']
    wavelengths = [float(wavelength) for wavelength in wavelengths]

    # Turn float to int
    wavelengths = [int(wavelength) for wavelength in wavelengths]


    # Load camera matrix
    mtx = np.load(mtx_path)
    dist = np.load(dist_path)

    # Undistort the hypercube
    h, w = hcube.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Undistort
    dst = cv.undistort(hcube, mtx, dist, None, newcameramtx)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Normalize hypercube
    dst = dst / np.max(dst)

    return dst, wavelengths


# Show images
def previewSnapshot(img, wavelengths, selected_pixel, selected_spectrum):

    # Print array as image
    plt.figure()
    plt.imshow(img[:, :, selected_spectrum])
    plt.plot(selected_pixel[1], selected_pixel[0], 'ro')
    plt.title('Image at Wavelength ' + str(wavelengths[selected_spectrum]) + 'nm')
    plt.show()

    # Print spectral composition of a single pixel
    spectrum = img[selected_pixel[0], selected_pixel[1]]

    # Plot the spectrum
    plt.figure()
    plt.plot(wavelengths, spectrum)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Spectral Composition of Pixel ' + str(selected_pixel))
    plt.axvline(wavelengths[selected_spectrum], color='r', linestyle='-')
    plt.show()



# Full preprocessing of hrHSI file
def preprocessFullHSI(path_to_hdf5, mtx_path, dist_path, x_off, y_off, rot, shear, ss_shape, ss_wavelengths):
    # Open the HDF5 file
    with h5py.File(path_to_hdf5, 'r') as f:

        # get the dataset from the file (this is how we save the objects downstairs)
        dataset = f['hypercube']
        hcube = f['hypercube'][:]
        
        # # Normalize hypercube
        # hcube = hcube / np.max(hcube)

        # Flip hypercube to align with snapshot
        hcube = np.flip(hcube, axis=1)
        hcube = np.flip(hcube, axis=2)

        # Rearrange hypercube to be in the shape of (x, y, wavelength)
        hcube = np.moveaxis(hcube, 0, -1)

        # Load wavelengths as integers
        wavelengths = np.array(dataset.attrs['wavelength_nm']).astype(int)
        wavelengths = wavelengths.tolist()


        # Load camera matrix
        mtx = np.load(mtx_path)
        dist = np.load(dist_path)



        # Undistort the hypercube
        h, w = hcube.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        img = cv.undistort(hcube, mtx, dist, None, newcameramtx)

        # Crop the image
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]


        # Select only wavelenghts that appear in the snapshot
        wavelengths_idx = []
        for i in range(len(ss_wavelengths)):
            idx = (np.abs(np.array(wavelengths) - np.array(ss_wavelengths[i]))).argmin()
            wavelengths_idx.append(idx)

        # Store the selected wavelength values
        hwavelengths_filtered = [wavelengths[i] for i in wavelengths_idx]

        # Select the 24 wavelenghts
        img = img[:, :, wavelengths_idx]


        # Shear image
        tform = transform.AffineTransform(shear=shear)
        img = transform.warp(img, tform)

        # Rotate hrHSI
        img = ndimage.rotate(img, angle=rot, reshape=False)

        # Crop hrHSI to match snapshot
        img = img[y_off[0]:y_off[1], x_off[0]:x_off[1], :]

        # Stretch hrHSI to match snapshot aspect ratio
        new_y = ((ss_shape[0] / ss_shape[1]) * img.shape[1])
        img = transform.resize(img, (new_y, img.shape[1], img.shape[2]), order=1)

        # Normalize hypercube
        img = img / np.max(img)

        return img, hwavelengths_filtered
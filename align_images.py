import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform

def align_images(hr_img, ss_img, ss_x_off, ss_y_off, hr_x_off, hr_y_off, rot, shear):

    # Shear image
    tform = transform.AffineTransform(shear=shear)
    hr_img = transform.warp(hr_img, tform)

    # Rotate hrHSI
    hr_img = ndimage.rotate(hr_img, angle=rot, reshape=False)

    # Crop snapshot to match 1.9 ratio
    ss_img = ss_img[ss_y_off[0]:ss_y_off[1], ss_x_off[0]:ss_x_off[1], :]

    # Crop hrHSI to match snapshot
    hr_img = hr_img[hr_y_off[0]:hr_y_off[1], hr_x_off[0]:hr_x_off[1], :]

    # Stretch hrHSI to match snapshot aspect ratio
    new_y = ((ss_img.shape[0] / ss_img.shape[1]) * hr_img.shape[1])
    hr_img = transform.resize(hr_img, (new_y, hr_img.shape[1], hr_img.shape[2]), order=1)

    return hr_img, ss_img




def plot_image_comparison(hr_img, hr_wavelengths, ss_img, ss_wavelengths, selected_pixel, selected_spectrum):
    

    # Pick closest value to selected spectrum in list of wavelengths
    hr_selected_spectrum_index = hr_wavelengths.index(min(hr_wavelengths, key=lambda x:abs(x-selected_spectrum)))
    ss_selected_spectrum_index = ss_wavelengths.index(min(ss_wavelengths, key=lambda x:abs(x-selected_spectrum)))


    # Transform selected pixel to scale
    scaling = (hr_img.shape[0] / ss_img.shape[0], hr_img.shape[1] / ss_img.shape[1])
    selected_pixel_scaled = (selected_pixel[0]*scaling[0], selected_pixel[1]*scaling[1])

    # Plot hrHSI and snapshot image next to each other
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(hr_img[:, :, hr_selected_spectrum_index])
    plt.plot(selected_pixel_scaled[1], selected_pixel_scaled[0], 'ro')
    plt.title('High Resolution HSI' + ' at ' + str(hr_wavelengths[hr_selected_spectrum_index]) + 'nm')
    plt.subplot(1, 2, 2)
    plt.imshow(ss_img[:, :, ss_selected_spectrum_index])
    plt.plot(selected_pixel[1], selected_pixel[0], 'ro')
    plt.title('Snapshot' + ' at ' + str(ss_wavelengths[ss_selected_spectrum_index]) + 'nm')
    plt.show()



    # Compare the spectral composition of the selected pixel in both images
    plt.figure()
    plt.plot(hr_wavelengths, hr_img[int(selected_pixel_scaled[0]), int(selected_pixel_scaled[1]), :])
    plt.plot(ss_wavelengths, ss_img[selected_pixel[0], selected_pixel[1], :])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.ylim(0, 1)
    plt.title('Spectral Composition of Pixel ' + str(selected_pixel))
    plt.legend(['High Resolution HSI', 'Snapshot'])
    plt.axvline(selected_spectrum, color='r', linestyle='-')
    plt.show()
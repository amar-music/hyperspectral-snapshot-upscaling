import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def align_images(hr_img, ss_img, x_off, y_off, hr_rot, ss_rot):

    # Crop hrHSI to match snapshot
    hr_img_cropped = hr_img[y_off[0]:y_off[1], x_off[0]:x_off[1], :]

    # Rotate hrHSI
    hr_img_cropped_rotated = ndimage.rotate(hr_img_cropped, angle=hr_rot, reshape=False)

    # Rotate snapshot 
    ss_img_rotated = ndimage.rotate(ss_img, angle=ss_rot, reshape=True) # reshape=True to maintain resolution

    return hr_img_cropped_rotated, ss_img_rotated




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
    plt.title('Spectral Composition of Pixel ' + str(selected_pixel))
    plt.legend(['High Resolution HSI', 'Snapshot'])
    plt.axvline(selected_spectrum, color='r', linestyle='-')
    plt.show()
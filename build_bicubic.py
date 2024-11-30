import numpy as np
import glob
import scipy.ndimage

OPERATION = "downscale"


# Define bicubic function
def bicubic_upscale(input_file, output_file, scale_factor):
    # Load the hyperspectral image from the .npy file
    snapshot_image = np.load(input_file)
    
    # Perform bicubic upscaling
    upscaled_image = scipy.ndimage.zoom(snapshot_image, (scale_factor, scale_factor, 1), order=3)
    
    # Normalize the upscaled image to the range [0.0, 1.0]
    upscaled_image = (upscaled_image - np.min(upscaled_image)) / (np.max(upscaled_image) - np.min(upscaled_image))
    
    # Save the upscaled image to a new .npy file
    np.save(output_file, upscaled_image)
    print(f"Upscaled image saved to {output_file} with min: {upscaled_image.min()} and max: {upscaled_image.max()}")
    print(" ")



# Define bicubuc downscale function
def bicubic_downscale(input_file, output_file, scale_factor):
    # Load the hyperspectral image from the .npy file
    hr_image = np.load(input_file)
    
    # Perform bicubic downscaling
    downscaled_image = scipy.ndimage.zoom(hr_image, (1/scale_factor, 1/scale_factor, 1), order=3)
    
    # Normalize the downscaled image to the range [0.0, 1.0]
    downscaled_image = (downscaled_image - np.min(downscaled_image)) / (np.max(downscaled_image) - np.min(downscaled_image))
    
    # Save the downscaled image to a new .npy file
    np.save(output_file, downscaled_image)
    print(f"Downscaled image saved to {output_file} with min: {downscaled_image.min()} and max: {downscaled_image.max()}")
    print(" ")




if OPERATION == "upscale":

    # Specify paths
    ss_files = glob.glob("data/processed/full_hsi/test/lr/**")

    # Set goal size
    # goal_size = 240
    goal_size = 480
    scale_factor = goal_size / 210 


    for f in ss_files:
        print("Processing file: " + f + "... (" + str(ss_files.index(f)+1) + "/" + str(len(ss_files)) + ")")
        bicubic_upscale(f, "data/processed/full_hsi/test/bi_" + str(goal_size) + f[31:], scale_factor=scale_factor)


elif OPERATION == "downscale":

    # Specify paths
    hr_files = glob.glob("data/processed/full_hsi/hr/**")

    # Set goal size
    goal_size = 240
    scale_factor = 480 / goal_size


    for f in hr_files:
        print("Processing file: " + f + "... (" + str(hr_files.index(f)+1) + "/" + str(len(hr_files)) + ")")
        bicubic_downscale(f, "data/processed/full_hsi/synth_" + str(goal_size) + f[26:], scale_factor=scale_factor)

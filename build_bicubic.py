import numpy as np
import glob
import scipy.ndimage


# Define bicubic function
def bicubic_upscale(input_file, output_file, scale_factor):
    # Load the hyperspectral image from the .npy file
    snapshot_image = np.load(input_file)
    
    upscaled_image = scipy.ndimage.zoom(snapshot_image, (scale_factor, scale_factor, 1), order=3)
    
    # Save the upscaled image to a new .npy file
    np.save(output_file, upscaled_image)
    print(f"Upscaled image saved to {output_file}")



# Specify paths
ss_files = glob.glob("data/Processed/train/**")
scale_factor = 480/210
# scale_factor = 240/210 


for f in ss_files:
    print("Processing file: " + f + "... (" + str(ss_files.index(f)+1) + "/" + str(len(ss_files)) + ")")
    bicubic_upscale(f, "data/Processed/bicubic_480/" + f[21:], scale_factor=scale_factor)

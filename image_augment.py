import cv2
import numpy as np
from omegaconf import OmegaConf
import glob

config = OmegaConf.load('./params.yaml')
# define a functino that rotate the image
def image_rotation(imag, angle):
  #padding the image before rotation
  padding = config.image_aug.padding
  # Get the original image dimensions
  height, width, _ = imag.shape
  # Create a new image with padded dimensions
  padded_image = np.zeros((height + 2* padding, width + 2 * padding, 3), dtype=np.uint8)

  # Copy the original image to the center of the new image
  padded_image[padding:padding + height, padding:padding + width] = imag

  # Rotate the padded image
  center = (width // 2 + padding, height // 2 + padding)
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (padded_image.shape[1], padded_image.shape[0]))

  # Crop the rotated image back to the original size
  cropped_image = rotated_image[padding:padding + height, padding:padding + width]
  return cropped_image

def image_labelling(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  # Define the lower and upper bounds for the light blue color in HSV
  lower_light_blue = np.array([80, 50, 50])
  upper_light_blue = np.array([120, 255, 255])
  # Create a binary mask for the light blue color
  light_blue_mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
  # Apply the mask to the original image
  ret = np.sum(cv2.bitwise_and(image, image, mask=light_blue_mask), 2)
  ret[ret != 0] = 1
  result = ret
  
  # Define the lower and upper bounds for the red color in HSV
  lower_red = np.array([0, 100, 100])
  upper_red = np.array([10, 255, 255])
  # Create a binary mask for the red color
  mask = cv2.inRange(hsv, lower_red, upper_red)
  # Apply the mask to the original image
  ret = np.sum(cv2.bitwise_and(image, image, mask=mask), 2)
  ret[ret != 0] = 2
  result += ret
  
  # Define the lower and upper bounds for the green color in HSV
  lower_green = np.array([40, 40, 40])
  upper_green = np.array([80, 255, 255])
  # Create a binary mask for the green color
  mask = cv2.inRange(hsv, lower_green, upper_green)
  # Apply the mask to the original image
  ret = np.sum(cv2.bitwise_and(image, image, mask=mask), 2)
  ret[ret != 0] = 3
  result += ret
  

  # Define the lower and upper bounds for the purple color in HSV
  lower_purple = np.array([120, 50, 50])
  upper_purple = np.array([150, 255, 255])
  # Create a binary mask for the purple color
  purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
  # Apply the mask to the original image
  ret = np.sum(cv2.bitwise_and(image, image, mask=purple_mask), 2)
  ret[ret != 0] = 4
  result += ret
  return result

def image_augmentatation(inpt_image, mask_image, data_path):
  #==================================================================
  # Create training dataset by image augmentation
  # Rotate the image every {delta} degree 
  delta = config.image_aug.angle
  img_id = config.image_aug.train_start_id
  for angle in range(0, 360, delta):
    rotate_inpt = image_rotation(inpt_image, angle)
    cv2.imwrite(f'{data_path}train/i{img_id}.png', rotate_inpt)
    rotate_mask = image_rotation(mask_image, angle)
    cv2.imwrite(f'{data_path}train/m{img_id}.png', rotate_mask)
    rotate_labl = image_labelling(rotate_mask)
    np.save(f'{data_path}train/l{img_id}.npy', rotate_labl)
    img_id += 1
      
  # Flip the image horizontally and do the rotation
  flipped_horizontally_inpt = cv2.flip(inpt_image, 1)
  flipped_horizontally_mask = cv2.flip(mask_image, 1)
  for angle in range(0, 360, delta):
    rotate_inpt = image_rotation(flipped_horizontally_inpt, angle)
    cv2.imwrite(f'{data_path}train/i{img_id}.png', rotate_inpt)
    rotate_mask = image_rotation(flipped_horizontally_mask, angle)
    cv2.imwrite(f'{data_path}train/m{img_id}.png', rotate_mask)
    rotate_labl = image_labelling(rotate_mask)
    np.save(f'{data_path}/train/l{img_id}.npy', rotate_labl)
    img_id += 1
      
  #==================================================================    
  # Create test data: shift the image and rotate it
  angles = config.image_aug.test_angles
  shifts = config.image_aug.test_shift
  img_id = config.image_aug.test_start_id
  for angle, shift in zip(angles, shifts):
    # Define the shift values (positive values shift to the right, negative to the left)
    shift_x, shift_y = shift[0], shift[1]
    # Define the transformation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # Apply the translation (shift) to the test image
    inpt_shifted_image = cv2.warpAffine(inpt_image, M, (inpt_image.shape[1], inpt_image.shape[0]))
    rotate_inpt = image_rotation(inpt_shifted_image, angle)
    cv2.imwrite(f'{data_path}/test/i{img_id}.png', rotate_inpt)
    mask_shifted_image = cv2.warpAffine(mask_image, M, (mask_image.shape[1], mask_image.shape[0]))
    rotate_mask = image_rotation(mask_shifted_image, angle)
    cv2.imwrite(f'{data_path}/test/m{img_id}.png', rotate_mask)
    rotate_labl = image_labelling(rotate_mask)
    np.save(f'{data_path}/test/l{img_id}.npy', rotate_labl)
    img_id += 1

if __name__ == "__main__":
  path = config.image_aug.input_dir
  input_images = glob.glob(f'{path}input/i*.png')
  print(f"{len(input_images)} will be processed for augmentation.")
  for i in range(len(input_images)):
    image = cv2.imread(f'{path}input/i{i}.png')
    image = cv2.resize(image, (256,256))
    mask = cv2.imread(f'{path}input/m{i}.png')
    mask = cv2.resize(mask, (256,256))
    print("processing {i}th image...")
    image_augmentatation(image, mask, path)
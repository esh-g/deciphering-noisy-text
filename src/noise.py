import numpy as np 
import cv2 
import random as rd 

def gaussian_noise(img, sigma, rng, mean=0.0):
    img_f = img.astype(np.float32)
    noise = rng.normal(loc = mean, scale = sigma, size=img.shape).astype(np.float32)
    noisy = img_f+noise 
    return np.clip(noisy, 0, 255).astype(np.uint8)



def salt_and_pepper(img, amt, salt_ratio, rng):
    out = img.copy()
    u = rng.random(img.shape)
    
    salt_mask = u <amt*salt_ratio
    pepper_mask = (u>= amt*salt_ratio) & (u<amt)
    out[salt_mask] = 255
    out[pepper_mask] = 0 
    return out 

def speckle_noise(img, sigma, rng, mean=0.0):
    img_f = img.astype(np.float32)
    noise  = rng.normal(loc=mean, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img_f + img_f*noise 
    return np.clip(noisy, 0, 255).astype(np.uint8)

def rotate_skew(img, rng, theta_range=(-15, 15), scale_range=(0.9, 1.1),shear_x_range=(-0.15, 0.15), shear_y_range=(-0.15, 0.15)):
    theta=rng.uniform(*theta_range)
    scale=rng.uniform(*scale_range)
    shear_x=rng.uniform(*shear_x_range)
    shear_y=rng.uniform(*shear_y_range)
    height,width=img.shape[:2]
    rotation_matrix=cv2.getRotationMatrix2D((width/2, height/2), theta, scale)
    rotation_matrix=np.vstack([rotation_matrix, [0, 0, 1]])
    shear_matrix=np.array([[1, shear_x, 0],[shear_y, 1, 0],[0, 0, 1]]).astype(np.float64)
    translate_to_origin=np.array([[1, 0, -(width/2)],[0, 1, -(height/2)],[0, 0, 1]])
    translate_back=np.array([[1, 0, (width/2)],[0, 1, height/2],[0, 0, 1]])
    centred_shear=translate_back@shear_matrix@translate_to_origin
    transformation_matrix=rotation_matrix@centred_shear
    return cv2.warpAffine(img,transformation_matrix[:2, :],(width, height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
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

def jpeg_compression(img, rng, quality_range=(30, 95)):
    quality=rng.integers(quality_range[0], quality_range[1]+1)
    _,lowered_quality=cv2.imencode(".jpg",img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    output=cv2.imdecode(lowered_quality, cv2.IMREAD_UNCHANGED)
    return output

def elastic_warp(img,rng,sigma_range=(3,6),alpha_range=(20,80)):
    height, width = img.shape[:2]
    dx=rng.uniform(-1, 1, (height, width)).astype(np.float32)
    dy=rng.uniform(-1, 1, (height, width)).astype(np.float32)
    sigma=rng.uniform(*sigma_range)
    alpha=rng.uniform(*alpha_range)
    dx=cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma)
    dy=cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma)
    dx*=alpha
    dy*=alpha
    x, y=np.meshgrid(np.arange(width), np.arange(height))
    map_x=(x+dx).astype(np.float32)
    map_y=(y+dy).astype(np.float32)
    output=cv2.remap(img, map_x, map_y,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return output


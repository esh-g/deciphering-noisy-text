import numpy as np 
import cv2 
import random as rd 

def gaussian_noise(img, sigma, rng, mean=0.0):
    img_f = img.astype(np.float32)
    noise = rng.normal(loc = mean, scale = sigma, size=img.shape).astype(np.float32)
    noisy = img_f+noise 
    return np.clip(noisy, 0, 255).astype(np.unit8)



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
    return np.clip(noisy, 0, 255).astype(np.unit8)

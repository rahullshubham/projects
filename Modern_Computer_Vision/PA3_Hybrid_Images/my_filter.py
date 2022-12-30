import numpy as np
import cv2
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt 
import helpers


def my_imfilter(image, filter):
  (k,l) = (filter.shape)
  (m,n,c) = (image.shape) 

  filtered_image = np.zeros((m,n,c))  
  if (k % 2) == 0 or (l % 2) == 0: 
    raise Exception('my_imfilter function only accepts filters with odd dimensions') 

  offsetm = ((k-1)//2)
  offsetn = ((l-1)//2)
  npad = (offsetm, offsetm), (offsetn, offsetn), (0,0) 
  
  paddedimage = np.pad(image, (npad), 'reflect') 
  for n1 in range (n):
    for m1 in range(m):
      for c1 in range (c):
        filtered_image[m1,n1,c1] = (filter*paddedimage[m1:m1+k, n1:n1+l, c1]).sum()  

  return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):

  #print(image1.shape[0])
  #print(image2.shape[0])
  #assert image1.shape[0] == image2.shape[0]
  #assert image1.shape[1] == image2.shape[1]
  #assert image1.shape[2] == image2.shape[2]
  
  if image1.shape[0] != image2.shape[0] and image1.shape[1] != image2.shape[1]:
      image1 = cv2.resize(image1, dsize=(image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_CUBIC)
      
  
  s, k = cutoff_frequency, cutoff_frequency*2
  probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
  kernel = np.outer(probs, probs)

  low_frequencies = my_imfilter(image1, kernel)  
  high_frequencies = image2-my_imfilter(image2, kernel)  
  high_frequencies = np.clip(high_frequencies+0.5, 0.0, 1.0)
  hybrid_image = low_frequencies + high_frequencies  
  hybrid_image = np.clip(hybrid_image-0.5, 0.0, 1.0) 
  
  return low_frequencies, high_frequencies, hybrid_image

def run():
    image1 = helpers.load_image(r"C:\Users\rahgupt\Downloads\Assignment_3\Assignment_3\data\ex07\einstein.jpg")
    image2 = helpers.load_image(r"C:\Users\rahgupt\Downloads\Assignment_3\Assignment_3\data\ex07\marilyn.jpg")
    low, high,  hybrid = gen_hybrid_image(image2, image1, 5)
    output = helpers.vis_hybrid_image(hybrid)
    helpers.save_image(r"C:\Users\rahgupt\Downloads\Assignment_3\Assignment_3\data\ex07\experiment_swap.jpg", output)
    
    





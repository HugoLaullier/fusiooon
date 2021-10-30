import matplotlib.pyplot as plt
import numpy as np

def show_im(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0))) 

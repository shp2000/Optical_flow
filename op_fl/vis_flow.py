import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
        @threshmin: confidence must be greater than threshmin to be kept
    return value:
        None
    """

    """
    STUDENT CODE BEGINS
    """
    x=[]
    x= np.array(x)
    y=[]
    y = np.array(y)
    flow_x=[]
    flow_x=np.array(flow_x)
    flow_y=[]
    flow_y=np.array(flow_y)
    for ro in range(confidence.shape[0]):
        for co in range(confidence.shape[1]):
            if(confidence[ro][co]>threshmin):
                flow_x = np.append(flow_x, flow_image[ro][co][0])
                flow_y = np.append(flow_y,flow_image[ro][co][1])
                x=np.append(x,co)
                y=np.append(y, ro)



    
    """
    STUDENT CODE ENDS
    """
    
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, (flow_x*10).astype(int), (flow_y*10).astype(int), 
                    angles='xy', scale_units='xy', scale=1., color='red', width=0.001)
    
    return





    


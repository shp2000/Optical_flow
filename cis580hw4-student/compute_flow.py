import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """
    """
    STUDENT CODE BEGINS
    """
    # Ix=Ix.flatten()
    # Iy = Iy.flatten()
    # It = It.flatten()
    
    Ax=[]
    Ay = []
    At= []
    
    xlow = np.clip([x-(5//2)], 0, 511)
    xhigh = np.clip([x+(5//2)], 0, 511)
    ylow = np.clip([y-(5//2)], 0, 511)
    yhigh = np.clip([y+(5//2)], 0, 511)
    for i in range(ylow[0], yhigh[0]+1):
        for j in range(xlow[0],xhigh[0]+1):
            Ax.append(Ix[i][j])
            Ay.append(Iy[i][j])
            At.append(It[i][j])
    Ax = np.array(Ax)
    Ay = np.array(Ay)
    #print(len(Ax))
    Axy=np.zeros((len(Ax),2))
    for i in range(0,len(Ax)):
        Axy[i][0] = Ax[i]
        Axy[i][1] = Ay[i]
    Axy = np.array(Axy)

    At = np.array(At)
    Ax = np.reshape(Ax,(len(Ax),1))
    Ay = np.reshape(Ay,(len(Ay),1))
    At = np.reshape(At,(len(At),1))
    
    A,x,x,conf = np.linalg.lstsq(Axy, -At)
    # [U, S , Vt ] = np.linalg.svd (Axy)
    flow=[A[0][0],A[1][0]]
    conf = np.min(conf)

    
    """
    STUDENT CODE ENDS
    """
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

    


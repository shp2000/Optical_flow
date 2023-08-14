import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    params:
        @flow: np.array(h, w, 2)
        @confidence: np.array(h, w, 2)
        @K: np.array(3, 3)
        @ep: np.array(3,) the epipole you found epipole.py note it is uncalibrated and you need to calibrate it in this function!
    return value:
        depth_map: np.array(h, w)
    """
    depth_map = np.zeros_like(confidence)

    """
    STUDENT CODE BEGINS
    """
    u = flow[:,:,0]
    v = flow[:,:,1]
    u=u.flatten()
    v=v.flatten()
       
    xp = np.arange(0,512)
    yp = np.arange(0, 512)
    xp,yp=np.meshgrid(xp,yp)
    ze=np.zeros((1, len(u)))
    on = np.ones((1,len(u)))
    uv = np.vstack((u,v,ze))    
    
    
    
    xp = np.array(xp)
    yp = np.array(yp)
    xp = xp.flatten()
    yp = yp.flatten()
    xy= np.vstack((xp,yp,on))


    uv[0,:] = uv[0,:]/K[0][0]
    uv[1,:] = uv[1,:]/K[1][1]

    ep_cal = np.linalg.inv(K)@ep
    xy_cal = np.linalg.inv(K)@xy

    num = np.linalg.norm(xy_cal[:-1,:] - ep_cal[:-1].reshape(-1,1), axis=0)
    den = np.linalg.norm(uv, axis=0)
    depth = num/den

    depth_map = depth.reshape(depth_map.shape[0], depth_map.shape[1])
    depth_map = np.where(confidence>thres,depth_map,0)



    
    """
    STUDENT CODE ENDS
    """

    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    # You can change the depth bound for better visualization if your depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    # print(f'depth bound: {depth_bound}')

    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()
    

    return truncated_depth_map

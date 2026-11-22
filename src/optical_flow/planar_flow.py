import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
        @up: upper left index [i,j] of image region to consider.
        @down: lower right index [i,j] of image region to consider.
    return value:
        sol: np.array(8,)
    """
    """
    STUDENT CODE BEGINS
    """
    u = flow_x[up[0]:down[0], up[1]:down[1]]/K[0][0]
    v = flow_y[up[0]:down[0], up[1]:down[1]]/K[1][1]
    u=u.flatten()
    v=v.flatten()
    
    ze=np.zeros((1, len(u)))
    on = np.ones((1,len(u)))

    print(u.shape)
    print(u.reshape(-1,1).shape)

    #uv= np.vstack((u,v))
    uv = np.vstack((u.flatten().reshape(-1,1), v.flatten().reshape(-1,1)))


    xp = np.arange(0,512)
    yp = np.arange(0, 512)
    xp,yp=np.meshgrid(xp,yp)

    xp_ir = xp[up[0]:down[0], up[1]:down[1]]
    yp_ir = yp[up[0]:down[0], up[1]:down[1]]
    xp_ir=xp_ir.flatten()
    yp_ir=yp_ir.flatten()
    # xp_ir, yp_ir = np.meshgrid(xp_ir, yp_ir)

    xy_ir= np.vstack((xp_ir,yp_ir,on))
    xp_ir_cal = np.linalg.inv(K)@xy_ir
    print(xp_ir_cal.shape)

    x = np.hstack(((xp_ir_cal[0]**2).flatten().reshape(-1,1), (xp_ir_cal[0]*xp_ir_cal[1]).flatten().reshape(-1,1), xp_ir_cal[0].flatten().reshape(-1,1), xp_ir_cal[1].flatten().reshape(-1,1), np.ones(u.flatten().shape[0]).reshape(-1,1), np.zeros((u.flatten().shape[0], 3)) ))
    y = np.hstack(((xp_ir_cal[0]*xp_ir_cal[1]).flatten().reshape(-1,1), (xp_ir_cal[1]**2).flatten().reshape(-1,1), np.zeros((u.flatten().shape[0], 3)), xp_ir_cal[1].flatten().reshape(-1,1), xp_ir_cal[0].flatten().reshape(-1,1), np.ones(u.flatten().shape[0]).reshape(-1,1) ))
    x_y = np.vstack((x, y))
    print(uv.shape)
    print(y.shape)
    print(x_y.shape)
    sol = np.linalg.pinv(x_y)@uv
    #sol=np.array(sol)
    sol=np.reshape(sol,(8,))











    """
    STUDENT CODE ENDS
    """
    return sol
    

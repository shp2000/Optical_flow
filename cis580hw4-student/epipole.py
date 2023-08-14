import numpy as np
def epipole(u,v,smin,thresh,num_iterations = 1000):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold thresh 
    (for both sampling and finding inliers)
    params:
        @u: np.array(h,w)
        @v: np.array(h,w)
        @smin: np.array(h,w)
    return value:
        @best_ep: np.array(3,)
        @inliers: np.array(n,) 
    
    u, v and smin are (h,w), thresh is a scalar
    output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
    '''

    """
    You can do the thresholding on smin using thresh outside the RANSAC loop here. 
    Make sure to keep some way of going from the indices of the arrays you get below back to the indices of a flattened u/v/smin
    STUDENT CODE BEGINS
    """
   
    # u=np.array(u)
    # v=np.array(v)
    # smin=np.array(smin)
    u=u.flatten()
    v=v.flatten()
    smin = smin.flatten()

    xp = np.arange(-256,256)
    yp = np.arange(-256, 256)
    xp,yp=np.meshgrid(xp,yp)
    xp=np.array(xp)
    yp=np.array(yp)
    xp=xp.flatten()
    yp=yp.flatten()
    
    #u=u[smin>thresh]
    #v=v[smin>thresh]
    ze=np.zeros((len(xp),1))
    on = np.ones((len(u),1))
    uv = np.column_stack((u,v,ze))
    #print(uv.shape)(262144,3)
    
   # xp=xp[smin>thresh]
    #yp=yp[smin>thresh]
    
    xy= np.column_stack((xp,yp,on))
    #print(xy.shape)(262144,3)
    
    ind_thr = np.where(smin.flatten()>thresh)[0].flatten()  # valid indices
    #print(ind_thr.shape)(121762,)
    #print(ind_thr.flatten())

    sample_size = 2

    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations): #Make sure to vectorize your code or it will be slow! Try not to introduce a nested loop inside this one
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size] #indices for thresholded arrays you find above
        test_indices = permuted_indices[sample_size:] #indices for thresholded arrays you find above

        """
        STUDENT CODE BEGINS
        """
        # inliers = sample_indices
        sample = ind_thr[sample_indices]
        inliers = sample
        #print(sample.shape)(2,)
        #cr_pro = np.cross(uv[:,sample].T,xy[:,sample].T)
        #cr_pro = np.cross(uv[sample,:],xy[sample,:])
        cr_pro = np.cross(xy[sample,:],uv[sample,:])
        #print(cr_pro.shape)#(2,3)
        #print(xy[sample,:].shape)(2,3)
        U, S , Vt  = np.linalg.svd(cr_pro)
        ep = Vt[-1,:]
        #print(ep.shape) (3,)
        #print(len(et_T))

        test = ind_thr[test_indices]

        #dis = abs((np.cross(uv[:,test].T,xy[:,test].T)))@ep
        #dis = abs((np.cross(uv[test,:],xy[test,:])))@ep
        dis = abs((np.cross(xy[test,:], uv[test,:]))@ep  )
        inliers=np.append(inliers,test[dis<eps])
        # print(inliers)







        

        """
        STUDENT CODE ENDS
        """

        #NOTE: inliers need to be indices in flattened original input (unthresholded), 
        #sample indices need to be before the test indices for the autograder
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers
    print(best_ep) #0.94,0.31,-0

    return best_ep, best_inliers

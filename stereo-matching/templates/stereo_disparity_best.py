import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *


###########################################################
import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from imageio import imread, imwrite
from stereo_disparity_score import stereo_disparity_score
###########################################################


def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---

    # Parameters
    Lambda=maxd
    nb_iterations=120

    def compute_data_cost(I1, I2, maxd, Tau):
        """data_cost: a 3D array of sixe height x width x num_disp_value;
        data_cost(y,x,l) is the cost of assigning the label l to pixel (y,x).
        The cost is min(1/3 ||I1[y,x]-I2[y,x-l]||_1, Tau)."""
        h,w,_ = I1.shape
        dataCost=np.zeros((h,w,maxd))

        for lp in range(maxd):
            # print(I1 - np.roll(I2, lp, axis=1))
            # print(I1.shape, np.roll(I2, lp, axis=1).shape)
            dataCost[:, :, lp] = np.minimum(1./3*np.linalg.norm(I1 - np.roll(I2, lp, axis=1), axis=2, ord=1), Tau*np.ones((h, w)))

        return dataCost

    def compute_energy(dataCost,disparity,Lambda):
        """dataCost: a 3D array of sixe height x width x maxd;
        dataCost(y,x,l) is the cost of assigning the label l to pixel (y,x).
        disparity: array of size height x width containing disparity of each pixel.
        (an integer between 0 and maxd-1)
        Lambda: a scalar value.
        Return total energy, a scalar value"""
        h,w,maxd = dataCost.shape

        hh, ww = np.meshgrid(range(h), range(w), indexing='ij')
        dplp = dataCost[hh, ww, disparity]

        # Unitary cost of assigning this disparity to each pixel
        energy = np.sum(dplp)

        # Compute interaction cost of each neighbors
        interactionCostU = Lambda*(disparity - np.roll(disparity, 1, axis=0) != 0)
        interactionCostL = Lambda*(disparity - np.roll(disparity, 1, axis=1) != 0)
        interactionCostD = Lambda*(disparity - np.roll(disparity, -1, axis=0) != 0)
        interactionCostR = Lambda*(disparity - np.roll(disparity, -1, axis=1) != 0)

        # Ignoring edge costs
        interactionCostU[0, :] = 0
        interactionCostL[:, 0] = 0
        interactionCostD[-1, :] = 0
        interactionCostR[:, -1] = 0

        # Adding interaction cost of each neighbors
        energy += np.sum(interactionCostU)
        energy += np.sum(interactionCostL)
        energy += np.sum(interactionCostD)
        energy += np.sum(interactionCostR)

        return energy

    def update_msg(msgUPrev,msgDPrev,msgLPrev,msgRPrev,dataCost,Lambda):
        """Update message maps.
        dataCost: 3D array, depth=label number.
        msgUPrev,msgDPrev,msgLPrev,msgRPrev: 3D arrays (same dims) of old messages.
        Lambda: scalar value
        Return msgU,msgD,msgL,msgR: updated messages"""
        msgU=np.zeros(dataCost.shape)
        msgD=np.zeros(dataCost.shape)
        msgL=np.zeros(dataCost.shape)
        msgR=np.zeros(dataCost.shape)

        h,w,maxd = dataCost.shape

        msg_incoming_from_U = np.roll(msgDPrev, 1, axis=0)
        msg_incoming_from_L = np.roll(msgRPrev, 1, axis=1)
        msg_incoming_from_D = np.roll(msgUPrev, -1, axis=0)
        msg_incoming_from_R = np.roll(msgLPrev, -1, axis=1)

        npqU = dataCost + msg_incoming_from_L + msg_incoming_from_D + msg_incoming_from_R
        npqL = dataCost + msg_incoming_from_U + msg_incoming_from_D + msg_incoming_from_R
        npqD = dataCost + msg_incoming_from_L + msg_incoming_from_U + msg_incoming_from_R
        npqR = dataCost + msg_incoming_from_L + msg_incoming_from_D + msg_incoming_from_U

        spqU = np.amin(npqU, axis=2)
        spqL = np.amin(npqL, axis=2)
        spqD = np.amin(npqD, axis=2)
        spqR = np.amin(npqR, axis=2)

        for lp in range(maxd):
            msgU[:, :, lp] = np.minimum(npqU[:, :, lp], Lambda + spqU)
            msgL[:, :, lp] = np.minimum(npqL[:, :, lp], Lambda + spqL)
            msgD[:, :, lp] = np.minimum(npqD[:, :, lp], Lambda + spqD)
            msgR[:, :, lp] = np.minimum(npqR[:, :, lp], Lambda + spqR)

        return msgU,msgD,msgL,msgR

    def normalize_msg(msgU,msgD,msgL,msgR):
        """Subtract mean along depth dimension from each message"""

        avg=np.mean(msgU,axis=2)
        msgU -= avg[:,:,np.newaxis]
        avg=np.mean(msgD,axis=2)
        msgD -= avg[:,:,np.newaxis]
        avg=np.mean(msgL,axis=2)
        msgL -= avg[:,:,np.newaxis]
        avg=np.mean(msgR,axis=2)
        msgR -= avg[:,:,np.newaxis]

        return msgU,msgD,msgL,msgR

    def compute_belief(dataCost,msgU,msgD,msgL,msgR):
        """Compute beliefs, sum of data cost and messages from all neighbors"""
        beliefs=dataCost.copy()

        msg_incoming_from_U = np.roll(msgD, 1, axis=0)
        msg_incoming_from_L = np.roll(msgR, 1, axis=1)
        msg_incoming_from_D = np.roll(msgU, -1, axis=0)
        msg_incoming_from_R = np.roll(msgL, -1, axis=1)

        beliefs += msg_incoming_from_D + msg_incoming_from_L + msg_incoming_from_R + msg_incoming_from_U

        return beliefs

    def MAP_labeling(beliefs):
        """Return a 2D array assigning to each pixel its best label from beliefs
        computed so far"""
        return np.argmin(beliefs, axis=2)

    def stereo_bp(I1,I2,maxd,Lambda,Tau=15,num_iterations=60):
        """The main function"""
        dataCost = compute_data_cost(I1, I2, maxd, Tau)
        energy = np.zeros((num_iterations)) # storing energy at each iteration
        # The messages sent to neighbors in each direction (up,down,left,right)
        h,w,_ = I1.shape
        msgU=np.zeros((h, w, maxd))
        msgD=np.zeros((h, w, maxd))
        msgL=np.zeros((h, w, maxd))
        msgR=np.zeros((h, w, maxd))

        print('Iteration (out of {}) :'.format(num_iterations))
        for iter in range(num_iterations):
            print('\t'+str(iter))
            msgU,msgD,msgL,msgR = update_msg(msgU,msgD,msgL,msgR,dataCost,Lambda)
            msgU,msgD,msgL,msgR = normalize_msg(msgU,msgD,msgL,msgR)
            # Next lines unused for next iteration, could be done only at the end
            beliefs = compute_belief(dataCost,msgU,msgD,msgL,msgR)
            disparity = MAP_labeling(beliefs)
            energy[iter] = compute_energy(dataCost,disparity,Lambda)

        return disparity,energy

    def to_rgb(im):
        # as 1, but we use broadcasting in one line
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, :] = im[:, :, np.newaxis]
        return ret


    # Convert as float gray images
    Il = to_rgb(Il)
    Ir = to_rgb(Ir)
    Il=Il.astype(float)
    Ir=Ir.astype(float)

    # Gaussian filtering
    I1=gaussian_filter(Il, 0.6)
    I2=gaussian_filter(Ir, 0.6)

    Id,energy = stereo_bp(I1,I2,maxd,Lambda, num_iterations=nb_iterations)
    imwrite('disparity_{:g}.png'.format(Lambda),Id)

    #------------------

    return Id


###################################################################
if __name__ == '__main__':
    bboxes = loadmat("../images/bboxes.mat")

    # Load the stereo images.
    Il = imread("../images/cones_image_02.png", as_gray = True)
    Ir = imread("../images/cones_image_06.png", as_gray = True)
    It = imread("../images/cones_disp_02.png", as_gray = True)
    bbox = np.array(bboxes["cones_02"]["bbox"])

    # Il = imread("../images/teddy_image_02.png", as_gray = True)
    # Ir = imread("../images/teddy_image_06.png", as_gray = True)
    bbox = np.array(bboxes["teddy_02"]["bbox"])

    Id = stereo_disparity_best(Il, Ir, bbox, 52)
    print("Score:", stereo_disparity_score(It, Id, bbox))
    plt.imshow(Id, cmap = "gray")
    plt.show()
###################################################################

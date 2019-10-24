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
    # http://courses.csail.mit.edu/18.337/2010/projects/reports/Kaneva_final_report.pdf


    # Parameters that require tuning ###########################################################################
    # Parameters
    maxiter = int(6000/maxd)-3 # max number of belief propagation iterations
    tau = 15. # for the data cost
    lamb = 1 # the impact of smoothness in the cost function
    model = "Potts"# "Linear" # model for the smoothness cost # Potts turned out to be better at higher iteration (i.e. iter=100/tau=15)
    K = 2. # only when using the truncated linear model fpr smoothness
    sigma = 0.1 # gaussian pamareter for the input images before stereo matching

    # Preprocessing
    print("Processing LBP with {} iterations, tau={} \nand lamb={} with the {} model for smoothness:".format(maxiter, tau, lamb, model))
    m, n = Il.shape # m=height(#rows) and n=width(#columns)
    Il, Ir = gaussian_filter(Il, sigma), gaussian_filter(Ir, sigma) # smoothen the img


    # Helper functions used in the Loopy Belief Propagation #####################################################
    def pass_messages(messages, direction="from"):
        """
        Helper function, takes in a bundle of 4 messages and pass to/from the up/down/left/right neighbours

        Parameters:
        -----------
        messages    - 4 x (m x n x maxd) or 4 x (m x n), a list of the messages of thr 4 neighbours, to be updated
                      note that an intensity map which doesn't have the "maxd" dimension can also be processed with it
        direction   - "from" or "to", specifying if the rolling is in forward (to) or backward (from) direction
                      in most of the cases such as belief and message propagation, we only need "from"

        Returns:
        --------
        messages    - Same as input dimension, the "rolled" messages
        """
        if direction == "from":
            msg_down, msg_up,  msg_right, msg_left = messages
        else:
            msg_up, msg_down, msg_left, msg_right = messages
        msg_up = np.roll(msg_up, 1, axis=0)
        msg_down = np.roll(msg_down, -1, axis=0)
        msg_left = np.roll(msg_left, 1, axis=1)
        msg_right = np.roll(msg_right, -1, axis=1)

        return msg_up, msg_down, msg_left, msg_right

    def data_cost_table(Il, Ir, maxd=maxd, tau=tau):
        """
        Helper function, computes the truncated absolute intensity difference
        as the cost of assigning disparity d to pixel p
        It is used as a D_cost lookup table for a given disparity mapping

        Parameters:
        -----------
        Il, Ir  - m x n x 1 (broadcasted)
        maxd    - maximum disparity value; disparities d's are taken in range(0, maxd)

        Returns:
        --------
        D_costs - m x n x maxd, the data cost d_cost for each disparity d, on each pixel p
                  where d_cost = min(|Il(y,x)-Ir[y,x-d]|, tau)
        """
        D_costs = np.zeros((m, n, maxd))

        for d in range(maxd):
            # Along x direction, shift each pixel by d to get Ir[y,x-d]
            # Note that np.roll is used to prevent values beyound x's range
            # Alternatively, we can do a subtraction and then turn values <0 or >n-1 into 0 and n-1
            Ir_shifted = np.roll(Ir, d, axis=1)
            D_costs[:, :, d] = np.minimum(np.abs(Il - Ir_shifted), tau*np.ones((m, n)))
        return D_costs

    def smoothness_cost(Id1, Id2, neighbour, model="Potts", lamb=lamb):
        """
        Helper function, computes the smoothness cost given two neighbours

        Parameters:
        -----------
        Id1     - m x n, disparity mapping 1
        Id2     - m x n, disparity mapping 2
        neighbour   - string, specifying which neightbour it is
        model   - string, either "Potts" or "Linear"
        lamb    - scaler, smoothness parameter,the bigger, the smoother

        Returns:
        --------
        S_cost  - scaler, sum of all S(p1-p2) = {p1=p2: 0, otherwise: 1}
        """
        # Get smoothness cost of the neighbour in each direction by the Potts or truncated linear model
        if model == "Potts":
            S_cost = (Id1 - Id2)!=0
        else:
            S_cost = np.minimum(np.abs(Id1 - Id2), K*np.ones((m, n)))

        # The extra edge "rolled" by the shift is turned back to its original values
        if neighbour == 'up':
            S_cost[0, :] = 0
        elif neighbour == 'down':
            S_cost[-1, :] = 0
        elif neighbour == 'left':
            S_cost[:, 0] = 0
        else:
            S_cost[:, -1] = 0

        return S_cost * float(lamb)

    def smoothness_cost_total(Id, model="Potts", lamb=lamb):
        """
        Helper function, computes the total smoothness cost (i.e. left/right/up/down) at given Id map
        Note that there are many different models, and here the Potts model us used to reduce computation

        Parameters:
        -----------
        Id      - m x n, disparity mapping
        model   - string, either "Potts" or "Linear"
        lamb    - scaler, smoothness parameter,the bigger, the smoother

        Returns:
        --------
        S_cost  - scaler, sum of all S(p1-p2) = {p1=p2: 0, otherwise: 1}
        """
        # Disparity map of the current Id shifted by 1 in each direction
        Id_up, Id_down, Id_left, Id_right = pass_messages([Id, Id, Id, Id], direction='to')

        S_cost_up = smoothness_cost(Id, Id_up, 'up', model, lamb)
        S_cost_down = smoothness_cost(Id, Id_down, 'down', model, lamb)
        S_cost_left = smoothness_cost(Id, Id_left, 'left', model, lamb)
        S_cost_right = smoothness_cost(Id, Id_right, 'right', model, lamb)

        # Sum up the total cost across the image for each direction
        S_cost = (np.sum(S_cost_up)+
                  np.sum(S_cost_down)+
                  np.sum(S_cost_left)+
                  np.sum(S_cost_right))

        return S_cost

    def total_energy(Id, D_costs, lamb=lamb):
        """
        Helper function, returns the total energy at the current disparity mapping
        Note that this is the objective function to be minimized w.r.t. disparity (Id)!

        Parameters:
        -----------
        Id, D_costs, and lamb are the same inputs as the above functions

        Returns:
        --------
        energe  - scaler, sum of data cost and smoothness cost computed by the above helper functions
        """
        # Finds the individual data costs given the current disparity mapping, and add them together
        data_cost = 0
        for i in range(m):
            for j in range(n):
                data_cost += D_costs[i, j, Id[i, j]]

        energy = data_cost + smoothness_cost_total(Id, model="Potts")

        return energy

    def propagate(messages, D_costs, lamb=lamb, model="Potts"):
        """
        Helper function, computes the total smoothness cost (i.e. left/right/up/down) at given Id map
        Note that there are many different models, and here the Potts model us used to reduce computation

        Parameters:
        -----------
        messages - 4 x (m x n x maxd), a list of the messages of thr 4 neighbours, to be updated
        D_costs and lamb are the same as in all the other functions

        Returns:
        --------
        messages - The updated list of messages
        """

        # Messages pasaed from it's neighbours previously
        msg_up_prev, msg_down_prev, msg_left_prev, msg_right_prev = pass_messages(messages)

        # The data cost and previous messages, using the min-sum model
        energy_up = D_costs + msg_left_prev + msg_down_prev + msg_right_prev
        energy_left = D_costs + msg_up_prev + msg_down_prev + msg_right_prev
        energy_down = D_costs + msg_left_prev + msg_up_prev + msg_right_prev
        energy_right = D_costs + msg_left_prev + msg_down_prev + msg_up_prev
        min_energy_up = np.amin(energy_up, axis=2)
        min_energy_left = np.amin(energy_left, axis=2)
        min_energy_down = np.amin(energy_down, axis=2)
        min_energy_right = np.amin(energy_right, axis=2)

        # Initialize and update the new messages: new msg at this neighbour with Id1 is given by
        #   min{D(Id2) + S(Id1, Id2) + sum(previous msgs except this one itself)}
        msg_up,msg_down,msg_left,msg_right= np.zeros((m, n, maxd)), np.zeros((m, n, maxd)), \
                                            np.zeros((m, n, maxd)), np.zeros((m, n, maxd))
        for d in range(maxd):
            msg_up[:, :, d] = np.minimum(energy_up[:, :, d], min_energy_up + \
                              smoothness_cost(min_energy_up,msg_up_prev[:, :, d], 'up', model, lamb))
            msg_left[:, :, d] = np.minimum(energy_left[:, :, d], min_energy_left + \
                                smoothness_cost(min_energy_left,msg_left_prev[:, :, d], 'down', model, lamb))
            msg_down[:, :, d] = np.minimum(energy_down[:, :, d], min_energy_down + \
                                smoothness_cost(min_energy_down,msg_down_prev[:, :, d], 'left', model, lamb))
            msg_right[:, :, d] = np.minimum(energy_right[:, :, d], min_energy_right + \
                                 smoothness_cost(min_energy_right,msg_right_prev[:, :, d], 'right', model, lamb))

        # Normalize each message pack, so that they have 0 as mean, otherwise they'd all grow huge
        messages = [msg_up,msg_down,msg_left,msg_right]
        for idx, msg in enumerate(messages):
            mean = np.mean(msg, axis=2)
            messages[idx] -= mean[:, :, np.newaxis]
        return messages

    # The main body of Loopy Belief Propagation #################################################################
    # Precompute the D_cost table
    D_costs = data_cost_table(Il, Ir, maxd)

    # Initiate messages from up, down, left, and right
    # Use np.ones if sum-product is used insteaf of min-sum
    messages = [np.zeros((m, n, maxd))] * 4
    energy = []

    # Perform the LBP to minimize the energy
    for i in range(maxiter):
        print('Iteration {}'.format(i))
        messages = propagate(messages, D_costs, model=model)
        # energy.append(total_energy(Id, D_costs)) # used to debug and observe, commented out for saving time

    # Once the max number of iteration is reached, we get the belief from the updated messages
    msg_up, msg_down, msg_left, msg_right = pass_messages(messages)
    Beliefs = D_costs + msg_up + msg_down + msg_left + msg_right

    # The final guess of Id is the disparsity mapping which yields the least cost (i.e. Beliefs)
    Id = np.argmin(Beliefs, axis=2)
    #------------------

    return Id


###################################################################
if __name__ == '__main__':
    bboxes = loadmat("../images/bboxes.mat")

    # Load the stereo images.
    # Il = imread("../images/cones_image_02.png", as_gray = True)
    # Ir = imread("../images/cones_image_06.png", as_gray = True)
    # It = imread("../images/cones_disp_02.png", as_gray = True)
    # bbox = np.array(bboxes["cones_02"]["bbox"])

    Il = imread("../images/teddy_image_02.png", as_gray = True)
    Ir = imread("../images/teddy_image_06.png", as_gray = True)
    It = imread("../images/teddy_disp_02.png", as_gray = True)
    bbox = np.array(bboxes["teddy_02"]["bbox"])

    Id = stereo_disparity_best(Il, Ir, bbox, 63)
    print("Score:", stereo_disparity_score(It, Id, bbox))
    imwrite('Id_lamb{}_iter{}.png'.format(lamb, maxiter),Id)
    # plt.imshow(Id, cmap = "gray")
    # plt.show()
###################################################################

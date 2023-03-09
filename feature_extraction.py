import cv2
import numpy as np
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
from skimage import feature

########################################
## Code for Local Persistant Homology ##
########################################


# Betti Curves Function Initializations
CP0 = CubicalPersistence(
    homology_dimensions=[0],
    coeff=3,
    n_jobs=-1
)
CP1 = CubicalPersistence(
    homology_dimensions=[1],
    coeff=3,
    n_jobs=-1
)
CP10 = CubicalPersistence(
    homology_dimensions=[0, 1],
    coeff=3,
    n_jobs=-1
)
BC = BettiCurve(n_bins=100)


def betti0(img):
    '''
    :param img: RGB Image file
    :type img: nd-array
    :return: TDA features vector
    :rtype: nd-array of size (100, )
    '''
    im_gray_h1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diagram_h1_0 =CP0.fit_transform(np.array(img)[None, : , :])
    y_betti_curve = BC.fit_transform(diagram_h1_0)
    return np.reshape(y_betti_curve,100).tolist()


def betti1(img):
    '''
    :param img: RGB Image file
    :type img: nd-array
    :return: TDA features vector
    :rtype: nd-array of size (100, )
    '''
    diagram_h1_0 = CP1.fit_transform(np.array(img)[None, :, :])
    y_betti_curve = BC.fit_transform(diagram_h1_0)
    return np.reshape(y_betti_curve, 100).tolist()


def generate_betti_input(img):
    im_gray_h1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diagram_h1_0 = CP10.fit_transform(np.array(im_gray_h1)[None, :, :])
    y_betti_curves_gray = BC.fit_transform(diagram_h1_0)

    red_channel = img[:, :, 2]
    diagram_h1_1 = CP10.fit_transform(np.array(red_channel)[None, :, :])
    y_betti_curves_red = BC.fit_transform(diagram_h1_1)

    green_channel = img[:, :, 1]
    diagram_h1_2 = CP10.fit_transform(np.array(green_channel)[None, :, :])
    y_betti_curves_green = BC.fit_transform(diagram_h1_2)

    blue_channel = img[:, :, 0]
    diagram_h1_3 = CP10.fit_transform(np.array(blue_channel)[None, :, :])
    y_betti_curves_blue = BC.fit_transform(diagram_h1_3)

    return np.reshape(y_betti_curves_gray, 200).tolist() + \
           np.reshape(y_betti_curves_red, 200).tolist() + \
           np.reshape(y_betti_curves_green, 200).tolist() +\
           np.reshape(y_betti_curves_blue, 200).tolist()


####################################
## Code for Local Binary Patterns ##
####################################

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist.tolist()

desc = LocalBinaryPatterns(48, 5)

def generate_lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = desc.describe(gray)
    lbp = lbp + desc.describe(img[:, :, 2])
    lbp = lbp + desc.describe(img[:, :, 1])
    lbp = lbp + desc.describe(img[:, :, 0])
    return lbp


############################
## Code for Gabor Filters ##
############################

# Constants for feature extraction
gamma=0.5
sigma=0.56
theta_list=[0, np.pi, np.pi/2, np.pi/4, 3*np.pi/4]
phi=0
lamda_list=[0, np.pi, np.pi/2, np.pi/4, 3*np.pi/4]


def gabor_features(img):
    local_energy_list = []
    mean_ampl_list = []
    for theta in theta_list:
        for lamda in lamda_list:
            kernel = cv2.getGaborKernel((3, 3), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
            fimage = cv2.filter2D(img, cv2.CV_8UC3, kernel)
            mean_ampl = np.sum(abs(fimage))
            mean_ampl_list.append(mean_ampl)

            local_energy = np.sum(fimage ** 2)
            local_energy_list.append(local_energy)
    return local_energy_list + mean_ampl_list


def generate_gabor_filters(image):
    return gabor_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))+gabor_features(image[:, 2])+gabor_features(image[:, 1])+gabor_features(image[:, 0])
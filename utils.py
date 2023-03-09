import cv2
from skimage import filters

def time_break(seconds):
    '''
    Funtion to generate Hrs, Mins, Sec break from seconds
    :param seconds: Input Seconds
    :return: Hrs, Mins and Seconds
    '''
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(((seconds % 3600) % 60))
    return hrs, mins, secs

def tile_content(img):
    '''
    Function to return While tile content values
    :param img: Input tile
    :return: White content Value
    '''
    warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    T = filters.threshold_otsu(warped)
    warped2 = (warped > T).astype("uint8")
    count = warped2.sum()
    perc = round(count / (warped.shape[0] * warped.shape[1]), 2)
    return perc
from utils.utils import compute_size
import sys
import cv2
import numpy as np
sys.path.append('../utils')

@compute_size
def exposure_fusion(img_list: list) -> np.ndarray:
    """applies exposure fusion to a list of images with different exposures

    Args:
        img_list (list): list of images

    Returns:
        np.ndarray: 8 bit LDR image
    """
    mergeMertens = cv2.createMergeMertens()
    resFusion = mergeMertens.process(img_list)
    return np.clip(resFusion*255, 0, 255).astype('uint8')


@compute_size
def align_images(img_list: list, **kwargs):

    cv2.createAlignMTB().process(img_list, img_list)
    print("Images aligned")
    return img_list


@compute_size
def calibrate_debevec(img_list: list, exposure_times: list):
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(img_list, exposure_times)
    print("Calibrated")
    return responseDebevec


@compute_size
def merge_debevec(img_list: list, exposure_times: list, response=None):
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(img_list, exposure_times, response)
    print("Merged")
    return hdrDebevec

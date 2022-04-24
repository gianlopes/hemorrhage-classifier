# from https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage
import numpy as np
import pydicom

def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)

def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value

def get_dicom_raw(dicom):
    return {attr:cast(getattr(dicom,attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}

def rescale_image(image, slope, intercept, bits, pixel):
    # In some cases intercept value is wrong and can be fixed
    # Ref. https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
    if bits == 12 and pixel == 0 and intercept > -100:
        image = image.copy() + 1000
        px_mode = 4096
        image[image>=px_mode] = image[image>=px_mode] - px_mode
        intercept = -1000

    return image.astype(np.float32) * slope + intercept # for translation adjustments given in the dicom file. 

def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2 # minimum HU level
    max_value = center + width // 2 # maximum HU level
    image[image < min_value] = min_value # set img_min for all HU levels less than minimum HU level
    image[image > max_value] = max_value # set img_max for all HU levels higher than maximum HU level
    return image

def get_windowed_ratio(image, center, width):
    # get ratio of pixels within the window
    windowed = apply_window(image, center, width)
    return len(np.where((windowed > 0) & (windowed < 80))[0]) / windowed.size

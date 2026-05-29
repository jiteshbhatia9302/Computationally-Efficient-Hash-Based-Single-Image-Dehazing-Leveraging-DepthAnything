import numpy as np

def histogram_equalization(channel):
    img = (channel * 255).astype(np.uint8)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    z = cdf_masked.min()
    cdf_eq = ((cdf_masked - z) * 255 / (cdf_masked.max() - z))
    cdf_eq = np.ma.filled(cdf_eq, 0).astype('uint8')
    equalized = cdf_eq[img]
    return equalized.astype(np.float32) / 255.0
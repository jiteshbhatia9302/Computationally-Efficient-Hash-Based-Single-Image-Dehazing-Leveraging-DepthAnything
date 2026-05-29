import numpy as np

def atmospheric_scattering_model(image_rgb, Gx, C=0.1):
    image = image_rgb.astype(np.float32) / 255.0
    J = Gx * (image - 1.0) + C
    J = np.clip(J, 0, 1)
    return J
import numpy as np
def compute_concatenated_vector(image_rgb, A, transmission_map, C=0.1):
    image = image_rgb.astype(np.float32) / 255.0
    transmission_map = np.expand_dims(transmission_map, axis=2)
    numerator = ((image - A) / (transmission_map + 1e-8)) + (A - C)
    denominator = image - 1.0
    Gx = numerator / (denominator + 1e-8)
    return Gx
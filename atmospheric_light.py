import numpy as np

def global_average_pooling(feature_map):
    return np.mean(feature_map, axis=(0, 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def pointwise_convolution(x, weights):
    return x * weights


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def estimate_atmospheric_light_CA(image_rgb):
    image = image_rgb.astype(np.float32) / 255.0
    fx = image
    gap = global_average_pooling(fx)
    weights1 = np.array([0.9, 0.8, 0.85])
    pw1 = pointwise_convolution(gap, weights1)
    gelu_output = gelu(pw1)
    weights2 = np.array([1.1, 1.0, 1.05])
    Z = pointwise_convolution(gelu_output, weights2)
    CA_c = sigmoid(Z)
    A = gap * CA_c
    return A
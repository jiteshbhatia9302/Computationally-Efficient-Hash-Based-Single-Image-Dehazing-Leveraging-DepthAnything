import numpy as np

def rgb_to_hsi(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-8
    theta = np.arccos(np.clip(numerator / denominator, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-8)) * min_rgb
    I = (R + G + B) / 3.0
    return np.dstack((H, S, I))


def hsi_to_rgb(hsi):
    H = hsi[:, :, 0] * 2 * np.pi
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    idx = (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / (np.cos(np.pi / 3 - H[idx]) + 1e-8))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])
    idx = ((H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3))
    H2 = H[idx] - 2 * np.pi / 3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H2)) / (np.cos(np.pi / 3 - H2) + 1e-8))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])
    idx = (H >= 4 * np.pi / 3)
    H3 = H[idx] - 4 * np.pi / 3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H3)) / (np.cos(np.pi / 3 - H3) + 1e-8))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])
    rgb = np.dstack((R, G, B))
    rgb = np.clip(rgb, 0, 1)
    return rgb
import numpy as np
from color_space import (rgb_to_hsi, hsi_to_rgb)
from histogram_utils import (histogram_equalization)

def post_processing_enhancement(J):
    M, N, _ = J.shape
    mu_R = np.sum(J[:, :, 0]) / (M * N)
    mu_G = np.sum(J[:, :, 1]) / (M * N)
    mu_B = np.sum(J[:, :, 2]) / (M * N)
    mu = np.array([mu_R, mu_G, mu_B])
    D_col = J - mu
    D_col_norm = (D_col - D_col.min()) / (D_col.max() - D_col.min() + 1e-8)
    hsi = rgb_to_hsi(D_col_norm)
    H = hsi[:, :, 0]
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]
    I_eq = histogram_equalization(I)
    I_enhanced = (I_eq - I_eq.min()) / (I_eq.max() - I_eq.min() + 1e-8)
    hsi_enhanced = np.dstack((H, S, I_enhanced))
    enhanced_rgb = hsi_to_rgb(hsi_enhanced)
    enhanced_rgb = np.clip(enhanced_rgb, 0, 1)
    return enhanced_rgb
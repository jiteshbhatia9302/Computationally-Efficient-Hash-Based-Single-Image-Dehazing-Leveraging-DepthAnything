import numpy as np
import hashlib

def sha256_hash(depth_value):
    return hashlib.sha256(str(depth_value).encode()).hexdigest()


def estimate_transmission_map(depth_map, beta=1.5):
    h, w = depth_map.shape
    transmission_map = np.zeros((h, w))
    hash_table = {}
    for i in range(h):
        for j in range(w):
            d = round(float(depth_map[i, j]), 4)
            hashed_key = sha256_hash(d)
            if hashed_key in hash_table:
                t = hash_table[hashed_key]
            else:
                t = np.exp(-beta * d)
                hash_table[hashed_key] = t
            transmission_map[i, j] = t
    transmission_map = np.clip(transmission_map, 0.05, 1.0)
    return transmission_map
from image_reader import read_image
from depth_estimation import estimate_depth_map
from atmospheric_light import estimate_atmospheric_light_CA
from transmission_estimation import estimate_transmission_map
from concatenated_vector import compute_concatenated_vector
from asm_model import atmospheric_scattering_model
from post_processing import post_processing_enhancement

import cv2
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------

image_path = "input.jpg"

# Step 1
image_rgb = read_image(image_path)

# Step 2
depth_map = estimate_depth_map(image_rgb)

# Step 3
A = estimate_atmospheric_light_CA(image_rgb)

# Step 4
transmission_map = estimate_transmission_map(depth_map)

# Step 5
Gx = compute_concatenated_vector(image_rgb, A, transmission_map)

# Step 6
J = atmospheric_scattering_model(image_rgb,Gx)

# Step 7
final_output = post_processing_enhancement(J)

# -------------------------------------------------------
# Save Results
# -------------------------------------------------------

cv2.imwrite(
    "final_output.jpg",
    cv2.cvtColor(
        (final_output * 255).astype("uint8"),
        cv2.COLOR_RGB2BGR
    )
)

# -------------------------------------------------------
# Display Results
# -------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title("Input")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(depth_map, cmap='inferno')
plt.title("Depth Map")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(transmission_map, cmap='gray')
plt.title("Transmission")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(final_output)
plt.title("Output")
plt.axis("off")

plt.show()
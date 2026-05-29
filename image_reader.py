import cv2

def read_image(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    return image_rgb
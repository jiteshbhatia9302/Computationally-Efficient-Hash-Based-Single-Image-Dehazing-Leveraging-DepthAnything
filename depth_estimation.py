import cv2
import torch
import numpy as np

from depth_anything.dpt import DepthAnything
from torchvision.transforms import Compose
from depth_anything.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet
)

def estimate_depth_map(image_rgb):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything.from_pretrained(
        'LiheYoung/depth_anything_vits14'
    ).to(device).eval()
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        PrepareForNet(),
    ])
    input_image = transform(
        {'image': image_rgb}
    )['image']
    input_tensor = torch.from_numpy(
        input_image
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = model(input_tensor)
    depth = torch.nn.functional.interpolate(
        depth[None],
        (image_rgb.shape[0], image_rgb.shape[1]),
        mode='bilinear',
        align_corners=False,
    )[0, 0]
    depth_map = depth.cpu().numpy()
    depth_map = (
        depth_map - depth_map.min()
    ) / (
        depth_map.max() - depth_map.min() + 1e-8
    )
    return depth_map
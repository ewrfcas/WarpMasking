from depth_warp import DepthWarp
from depth_anything_v2.dpt import DepthAnythingV2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# preset some hyper-parameters
device = "cuda"
weight_dtype = torch.bfloat16  # bf16 or fp16
image_path = "demo/temp.png"
h, w = 512, 512
# warp_pros = [0.4, 0.4, 0.1, 0.1, 0.]
warp_pros = [0., 1.0, 0., 0., 0.]  # the probabilities of different types of warping, including orbit left/right/up/down, pan left/right/up/down, forward/backward
rad_range = [15, 20]  # range of rotating range
init_focal_length = 500 / 512 * ((h + w) / 2)


def dino_normalize(image):
    # inputs:-1~1, return:imagenet normalize
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=image.device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=image.device).reshape(1, 3, 1, 1)
    image = (image + 1) / 2
    image = (image - mean) / std
    return image


if __name__ == '__main__':
    # initialize the depth model (depthanythingv2)
    depth_config = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    depth_model = DepthAnythingV2(**depth_config)
    depth_model.load_state_dict(torch.load(f'ckpts/depth_anything_v2_vitl.pth', map_location='cpu'))
    depth_model.eval()

    depth_model.to(device, dtype=weight_dtype)

    depthwarp = DepthWarp(device, init_focal_length=init_focal_length, probs=warp_pros, rad_range=rad_range)

    image = Image.open(image_path).convert('RGB')
    image = transforms.ToTensor()(image)[None] * 2 - 1  # [1,3,h,w] -1~1
    image = image.to(device)

    with torch.no_grad():
        with torch.autocast("cuda", enabled=True, dtype=weight_dtype):
            image_depth = F.interpolate(image, size=(518, 518), mode="bicubic")
            depth = depth_model(dino_normalize(image_depth)).float()  # [B,H,W]
        depth = torch.clamp_min(depth, 1e-5)
        depth = 1 / depth
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)
        warp_image, warp_mask, warp_depth = depthwarp.forward_depth_warp(image, depth, image.shape[0], h, w)
        back_mask = depthwarp.back_forward_warp(warp_image, warp_mask, warp_depth, origin_depth=depth)

    # saving
    forward_image = (warp_image + 1) / 2 * (1 - warp_mask)
    forward_image = transforms.ToPILImage()(forward_image[0])
    forward_image.save("demo/forward_image.png")

    forward_mask = transforms.ToPILImage()(warp_mask[0])
    forward_mask.save("demo/forward_mask.png")

    forward_backward_image = (image + 1) / 2 * (1 - back_mask)
    forward_backward_image = transforms.ToPILImage()(forward_backward_image[0])
    forward_backward_image.save("demo/forward_backward_image.png")

    forward_backward_mask = transforms.ToPILImage()(back_mask[0])
    forward_backward_mask.save("demo/forward_backward_mask.png")

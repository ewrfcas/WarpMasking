import copy
import numpy as np
import torch
from einops import rearrange
from kornia.geometry import PinholeCamera, transform_points, convert_points_from_homogeneous
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_rotation,
)
from softmax_splatting import softsplat
import torch.nn.functional as F
import kornia
import random
import torchvision.transforms.functional as TF


def get_random_value(values, range):  # values 0~1
    return values * (range[1] - range[0]) + range[0]


def sobel_filter(disp, mode="sobel", beta=10.0):
    sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
    sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
    alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

    return alpha


class DepthWarp(object):
    def __init__(self, device,
                 init_focal_length=500,
                 probs=[0.375, 0.375, 0.1, 0.1, 0.05],
                 dis_range=[0.9, 1.2],
                 rad_range=[3, 20],
                 walk_range=[0.3, 0.6],
                 w2c_multiply=1.5,
                 erode=[1],
                 dilate=[1, 3],
                 dilate_prob=[0.75, 0.25],
                 blur_rate=0.15,
                 gaussian_kernel=[3, 5, 7, 9, 11],
                 sigma=[4, 6],
                 depth_threshold=[0.05, 0.2],
                 sobel_beta=10.0,
                 sobel_threshold=0.3,
                 ill_posed_rate=0.1,
                 **kwargs):
        super().__init__()
        self.device = device
        self.init_focal_length = init_focal_length
        self.dis_range = dis_range
        self.rad_range = rad_range
        self.probs = probs
        self.warp_types = ["w2c_rounding", "c2w_rounding", "w2c_raise", "c2w_raise", "c2w_walking"]
        self.warp_type = None
        self.batch_size = None
        self.points = None
        self.depth = None
        self.w2c_multiply = w2c_multiply
        self.erode = erode
        self.dilate = dilate
        self.dilate_prob = dilate_prob
        self.sobel_beta = sobel_beta
        self.sobel_threshold = sobel_threshold
        self.blur_rate = blur_rate
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.depth_threshold = depth_threshold
        self.ill_posed_rate = ill_posed_rate
        self.walk_range = walk_range

    def convert_pytorch3d_kornia(self, camera, batch_size, h, w):
        R = camera.R
        T = camera.T
        extrinsics = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        extrinsics[:, :3, :3] = R
        extrinsics[:, :3, 3] = T
        K = torch.zeros((self.batch_size, 4, 4), device=self.device)
        K[:, 0, 0] = self.init_focal_length * (w / max(h, w))
        K[:, 1, 1] = self.init_focal_length * (h / max(h, w))
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        K[:, 2, 2] = 1
        K[:, 3, 3] = 1
        h = torch.tensor([h] * batch_size, device=self.device)
        w = torch.tensor([w] * batch_size, device=self.device)
        return PinholeCamera(K, extrinsics, h, w)

    def get_next_camera_w2c_round(self, first_view=False):
        # round along the camera
        r = get_random_value(torch.rand(size=(self.batch_size,), device=self.device), self.dis_range) * self.initial_median_depth

        next_camera = copy.deepcopy(self.predefined_camera)
        # 在这个视角下(w2c)，camera是原点，T为世界坐标系原点在camera下的坐标，center是旋转目标，以opencv坐标系而言在camera的前方
        # 世界坐标系将围绕center旋转，如果center=0,0,0， 那就说明camera本身就是旋转目标
        # center = r.unsqueeze(1).repeat(1, 3) * 1.5
        # center[:, :2] = 0
        center = torch.tensor([[0.0, 0.0, 0.0]]).reshape(1, 3).repeat(self.batch_size, 1).to(self.device)
        if first_view:
            theta = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
        else:
            theta = torch.deg2rad(get_random_value(torch.rand(size=(self.batch_size,)), self.rad_range) % 360)
            theta = theta * (torch.randint(0, 2, size=(self.batch_size,)) * 2 - 1)
            theta = theta.to(self.device)
        theta = theta * self.w2c_multiply  # w2c角度可以更大
        x = torch.sin(theta)
        y = torch.zeros_like(x)
        z = torch.cos(theta)
        r_vector = r.unsqueeze(1) * torch.stack([x, y, z], dim=1)  # [b,3]
        total_t = center + r_vector
        next_camera.T = total_t.to(self.device).float()
        next_camera.R = look_at_rotation(next_camera.T, at=center.tolist()).to(self.device)
        return next_camera

    def get_next_camera_c2w_round(self, first_view=False):
        next_camera = copy.deepcopy(self.predefined_camera)

        r = get_random_value(torch.rand(size=(self.batch_size,), device=self.device), self.dis_range) * self.initial_median_depth
        # world camera中心为原点
        if first_view:
            theta = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
        else:
            theta = torch.deg2rad(get_random_value(torch.rand(size=(self.batch_size,)), self.rad_range) % 360)
            theta = theta * (torch.randint(0, 2, size=(self.batch_size,)) * 2 - 1)
            theta = theta.to(self.device)
        # 逆时针旋转
        x = torch.sin(theta) * r
        y = torch.zeros_like(x)
        z = torch.cos(theta) * (-r)

        cam_pos = torch.stack([x, y, z], dim=1).to(self.device)
        R = look_at_rotation(cam_pos, at=[(0, 0, 0)] * self.batch_size, up=[(0, 1, 0)] * self.batch_size).to(self.device)
        c2w = torch.eye(n=4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        c2w[:, :3, :3] = R
        c2w[:, :3, 3:4] = cam_pos.reshape(-1, 3, 1)

        w2c = c2w.inverse()
        next_camera.T = w2c[:, :3, -1]
        next_camera.R = w2c[:, :3, :3]

        return next_camera

    def get_next_camera_w2c_raise(self, first_view=False):
        # round along the camera
        r = get_random_value(torch.rand(size=(self.batch_size,), device=self.device), self.dis_range) * self.initial_median_depth

        next_camera = copy.deepcopy(self.predefined_camera)
        # 在这个视角下(w2c)，camera是原点，T为世界坐标系原点在camera下的坐标，center是旋转目标，以opencv坐标系而言在camera的前方
        # 世界坐标系将围绕center旋转，如果center=0,0,0， 那就说明camera本身就是旋转目标
        # center = r.unsqueeze(1).repeat(1, 3) * 1.5
        # center[:, :2] = 0
        center = torch.tensor([[0.0, 0.0, 0.0]]).reshape(1, 3).repeat(self.batch_size, 1).to(self.device)
        if first_view:
            theta = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
        else:
            theta = torch.deg2rad(get_random_value(torch.rand(size=(self.batch_size,)), self.rad_range) % 360)
            theta = theta * (torch.randint(0, 2, size=(self.batch_size,)) * 2 - 1)
            theta = theta.to(self.device)
        theta = theta * self.w2c_multiply  # w2c角度可以更大

        y = torch.sin(-theta)
        x = torch.zeros_like(y)
        z = torch.cos(theta)
        r_vector = r.unsqueeze(1) * torch.stack([x, y, z], dim=1)  # [b,3]
        total_t = center + r_vector
        next_camera.T = total_t.to(self.device).float()
        next_camera.R = look_at_rotation(next_camera.T, at=center.tolist()).to(self.device)
        return next_camera

    def get_next_camera_c2w_raise(self, first_view=False):
        next_camera = copy.deepcopy(self.predefined_camera)

        r = get_random_value(torch.rand(size=(self.batch_size,), device=self.device), self.dis_range) * self.initial_median_depth
        # world camera中心为原点
        if first_view:
            theta = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
        else:
            theta = torch.deg2rad(get_random_value(torch.rand(size=(self.batch_size,)), self.rad_range) % 360)
            theta = theta * (torch.randint(0, 2, size=(self.batch_size,)) * 2 - 1)
            theta = theta.to(self.device)
        # 逆时针旋转
        y = torch.sin(-theta) * r
        x = torch.zeros_like(y)
        z = torch.cos(theta) * (-r)

        cam_pos = torch.stack([x, y, z], dim=1).to(self.device)
        R = look_at_rotation(cam_pos, at=[(0, 0, 0)] * self.batch_size, up=[(0, 1, 0)] * self.batch_size).to(self.device)
        c2w = torch.eye(n=4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        c2w[:, :3, :3] = R
        c2w[:, :3, 3:4] = cam_pos.reshape(-1, 3, 1)

        w2c = c2w.inverse()
        next_camera.T = w2c[:, :3, -1]
        next_camera.R = w2c[:, :3, :3]

        return next_camera

    def get_next_camera_c2w_walking(self, first_view=False):
        next_camera = copy.deepcopy(self.predefined_camera)

        r = get_random_value(torch.rand(size=(self.batch_size,), device=self.device), self.dis_range) * self.initial_median_depth
        # world camera中心为原点
        if first_view:
            move = torch.ones((self.batch_size,), device=self.device, dtype=torch.float32)
        else:
            move = get_random_value(torch.rand(size=(self.batch_size,), device=self.device), self.walk_range)
        x = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
        y = torch.zeros((self.batch_size,), device=self.device, dtype=torch.float32)
        z = (-r) * move

        cam_pos = torch.stack([x, y, z], dim=1).to(self.device)
        R = look_at_rotation(cam_pos, at=[(0, 0, 0)] * self.batch_size, up=[(0, 1, 0)] * self.batch_size).to(self.device)
        c2w = torch.eye(n=4, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        c2w[:, :3, :3] = R
        c2w[:, :3, 3:4] = cam_pos.reshape(-1, 3, 1)

        w2c = c2w.inverse()
        next_camera.T = w2c[:, :3, -1]
        next_camera.R = w2c[:, :3, :3]

        return next_camera

    def warp_splatting(self, images, depth, batch_size, h, w):
        # images should be -1~1 [b,c,h,w]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        self.depth = depth
        current_camera = self.convert_pytorch3d_kornia(self.current_camera, batch_size, h, w)  # w2c
        next_camera = self.convert_pytorch3d_kornia(self.target_camera, batch_size, h, w)  # w2c

        x = torch.arange(w)
        y = torch.arange(h)
        points = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
        points = rearrange(points, "h w c -> (h w) c").to(self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        points_3d = current_camera.unproject(points, rearrange(depth, "b c h w -> b (h w) c"))
        P = next_camera.intrinsics @ next_camera.extrinsics

        transformed_points = transform_points(P, points_3d)
        self.proj_xyz = rearrange(transformed_points, "b (h w) c -> b c h w", h=h, w=w)
        # target_cam_points3d = transform_points(next_camera.extrinsics, points_3d)
        transformed_z = transformed_points[..., [2]]
        self.points_2d = convert_points_from_homogeneous(transformed_points)

        flow = self.points_2d - points
        flow_tensor = rearrange(flow, "b (h w) c -> b c h w", w=w, h=h)
        importance = 1.0 / (transformed_z)
        importance_min = importance.amin(keepdim=True)
        importance_max = importance.amax(keepdim=True)
        weights = (importance - importance_min) / (importance_max - importance_min + 1e-6) * 20 - 10
        weights = rearrange(weights, "b (h w) c -> b c h w", w=w, h=h)
        warp_masks = torch.ones((batch_size, 1, h, w), dtype=torch.float32, device=self.device)
        boundaries_mask = self.get_boundaries_mask(1 / depth)

        input_data = torch.cat([images, warp_masks, boundaries_mask, 1 / depth], dim=1)  # [b,c',h,w]
        output_data = softsplat.softsplat(tenIn=input_data, tenFlow=flow_tensor, tenMetric=weights.detach(), strMode="soft").detach()
        warp_images = output_data[:, 0:3]
        warp_masks = output_data[:, 3:4]

        nans = warp_masks.isnan()
        warp_masks[nans] = 0
        warp_masks[warp_masks > 0] = 1
        warp_masks = 1 - warp_masks

        boundaries_mask = output_data[:, 4:5]
        nans = boundaries_mask.isnan()
        boundaries_mask[nans] = 0
        warp_depth_masks = warp_masks.clone()  # depth mask do not need boundary mask
        warp_masks = torch.maximum(warp_masks, boundaries_mask)

        warp_masks[warp_masks > 0.3] = 1
        warp_masks[warp_masks <= 0.3] = 0
        warp_depth_masks[warp_depth_masks > 0.3] = 1
        warp_depth_masks[warp_depth_masks <= 0.3] = 0

        warp_images = warp_images * (1 - warp_masks) + torch.ones_like(warp_images) * warp_masks * (-1)
        warp_depth = torch.clamp(1 / output_data[:, 5:6], 0, 1e5)
        # this is a trick to pad depth with zero
        warp_depth = warp_depth * (1 - warp_depth_masks) + torch.zeros_like(warp_depth) * warp_depth_masks

        return warp_images, warp_masks, warp_depth

    def forward_depth_warp(self, images, depth, batch_size, H, W):
        self.batch_size = batch_size
        self.predefined_camera = self.get_init_camera(batch_size, H, W)
        depth_filter_sky = depth.clone()
        depth_filter_sky = depth_filter_sky.reshape(batch_size, -1)
        self.initial_median_depth = []
        for i in range(depth_filter_sky.shape[0]):
            # hardcode to ignore very far scenes
            v = torch.median(depth_filter_sky[i, depth_filter_sky[i] < 0.008])
            if v.isnan():
                v = torch.median(depth_filter_sky[i, depth_filter_sky[i] < 0.02])
            if v.isnan():
                v = torch.median(depth_filter_sky[i, depth_filter_sky[i] < 0.1])
            if v.isnan():
                v = torch.median(depth_filter_sky[i, depth_filter_sky[i] < 1e3])
            if v.isnan():
                v = torch.median(depth_filter_sky[i, depth_filter_sky[i] < 1e5])
            if v.isnan():
                v = torch.median(depth_filter_sky[i])
            self.initial_median_depth.append(v)
        self.initial_median_depth = torch.stack(self.initial_median_depth)
        self.warp_type = np.random.choice(self.warp_types, size=1, p=self.probs, replace=False)[0]

        if self.warp_type == "w2c_rounding":  # w2c_rounding
            self.current_camera = self.get_next_camera_w2c_round(first_view=True)
            self.target_camera = self.get_next_camera_w2c_round()
        elif self.warp_type == "c2w_rounding":
            self.current_camera = self.get_next_camera_c2w_round(first_view=True)
            self.target_camera = self.get_next_camera_c2w_round()
        elif self.warp_type == "w2c_raise":
            self.current_camera = self.get_next_camera_w2c_raise(first_view=True)
            self.target_camera = self.get_next_camera_w2c_raise()
        elif self.warp_type == "c2w_raise":
            self.current_camera = self.get_next_camera_c2w_raise(first_view=True)
            self.target_camera = self.get_next_camera_c2w_raise()
        elif self.warp_type == "c2w_walking":
            self.current_camera = self.get_next_camera_c2w_walking(first_view=True)
            self.target_camera = self.get_next_camera_c2w_walking()
        else:
            raise NotImplementedError

        return self.warp_splatting(images, depth, batch_size, H, W)

    def back_forward_warp(self, warp_images, warp_masks, warp_depth, origin_depth=None):

        if random.random() < self.ill_posed_rate and origin_depth is not None:
            # 一定概率走较为复杂的back_warp_mask
            return self.back_forward_warp_ill_posed(warp_images, warp_masks, origin_depth)

        B, _, H, W = warp_images.shape
        back_grid = self.points_2d.reshape(B, H, W, 2)
        back_grid[..., 0] = back_grid[..., 0] / W * 2 - 1
        back_grid[..., 1] = back_grid[..., 1] / H * 2 - 1
        back_depth = F.grid_sample(warp_depth, grid=back_grid, mode="nearest", padding_mode="zeros")
        back_depth_mask = back_depth == 0
        back_depth[back_depth_mask] = 0
        # 有遮挡的区域origin depth的z会大于back depth
        depth_threshold = get_random_value(random.random(), self.depth_threshold)
        proj_mask = ((self.depth - back_depth.reshape(B, 1, H, W)) / self.depth) > depth_threshold
        proj_mask = rearrange(proj_mask, "b c h w -> b h w c").repeat(1, 1, 1, 2)
        back_grid[proj_mask] = -10000.0
        back_masks = 1 - F.grid_sample(1 - warp_masks, grid=back_grid, mode="nearest", padding_mode="zeros")
        back_masks[back_masks > 0] = 1

        if random.random() < self.blur_rate:
            back_masks = 1 - F.max_pool2d(1 - back_masks, kernel_size=3, stride=1, padding=1)
            kernel = np.random.choice(self.gaussian_kernel, size=1)[0]
            back_masks = F.max_pool2d(back_masks, kernel_size=max(kernel, 7), stride=1, padding=max(kernel, 7) // 2)
            sigma = get_random_value(random.random(), self.sigma)
            back_masks = TF.gaussian_blur(back_masks, kernel_size=[kernel, kernel], sigma=sigma)
            back_masks[back_masks < 0.5] = 0
            back_masks[back_masks >= 0.5] = 1
        else:
            dilate = int(np.random.choice(self.dilate, size=1, p=self.dilate_prob)[0])
            if dilate > 1:
                back_masks = F.max_pool2d(back_masks, kernel_size=dilate, stride=1, padding=dilate // 2)

        return back_masks

    def back_forward_warp_ill_posed(self, images, masks, depth):
        # 故意构造及其复杂的mask形式
        # mask: 0 is valid, 1 is invalid(masked)
        B, _, H, W = images.shape
        back_grid = self.points_2d.reshape(B, H, W, 2)  # 存在遮挡，重复赋值，depth小的区域应该置空
        grid_1d = back_grid[..., 1].long() * W + back_grid[..., 0].long()
        grid_1d = grid_1d.reshape(B, H * W)  # [B,HW]

        # depth要归一化到严格<1,因为idx最小单位间隔是1，不能改变idx顺序，
        # 由于depthanythingV2输出的非天空基本都是<0.1，所以直接clip即可
        depth_clip = torch.clip(depth, 0, 0.99).reshape(B, H * W)
        grid_depth_1d = grid_1d + depth_clip  # [B,HW]
        # 从小到大，所以相同位置第一个是depth最小的
        grid_1d_sorted, grid_1d_idx_sorted = torch.sort(grid_depth_1d, dim=1)
        grid_1d_sorted = grid_1d_sorted.long()  # 排序结束就不需要depth了
        # 错位相减 [1,1,2,3,3,3,4,4]->[1,1,2,3,3,3,4]-[1,2,3,3,3,4,4]=[0,-1,-1,0,0,-1,0]
        # 所有start_idx=0到end_idx+1均为重叠区域
        grid_1d_sorted_dislocated = grid_1d_sorted[:, :-1] - grid_1d_sorted[:, 1:]
        # [0,-1,-1,0,0,-1,0]->[1,0,0,1,1,0,1] # 第一个开始重叠的idx(==1)保留(depth最小), 0,3,6
        # 但是我们要去除的是其余重叠区域，所以真正要获取的是[1],[4,5],[7], 所以hw结果要+1
        condition_occlusion = (grid_1d_sorted_dislocated == 0).float()

        invalid_batch, invalid_hw = torch.where(condition_occlusion == 1)
        invalid_hw += 1
        back_grid = back_grid.reshape(B, H * W, 2)
        # 重新回到grid_1d_idx_sorted去取原来的grid index，并且将gather目标设为ood值
        back_grid[invalid_batch, grid_1d_idx_sorted[invalid_batch, invalid_hw]] = -10000.0
        back_grid = back_grid.reshape(B, H, W, 2)

        back_grid[..., 0] = back_grid[..., 0] / W * 2 - 1
        back_grid[..., 1] = back_grid[..., 1] / H * 2 - 1

        # back_images = F.grid_sample(images, grid=back_grid, mode="bilinear", padding_mode="zeros")
        back_masks = 1 - F.grid_sample(1 - masks, grid=back_grid, mode="nearest", padding_mode="zeros")
        back_masks[back_masks > 0] = 1
        # back_images = back_images * (1 - back_masks) + torch.ones_like(back_images) * back_masks * (-1)

        return back_masks

    def get_boundaries_mask(self, disparity):
        normalized_disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
        return sobel_filter(normalized_disparity, "sobel", beta=self.sobel_beta) < self.sobel_threshold

    def get_init_camera(self, batch_size, h, w):
        K = torch.zeros((batch_size, 4, 4), device=self.device)
        K[:, 0, 0] = self.init_focal_length * (w / max(h, w))
        K[:, 1, 1] = self.init_focal_length * (h / max(h, w))
        K[:, 0, 2] = w / 2
        K[:, 1, 2] = h / 2
        K[:, 2, 2] = 1
        K[:, 3, 3] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        T = torch.zeros((batch_size, 3), device=self.device)
        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((h, w),), device=self.device)
        return camera

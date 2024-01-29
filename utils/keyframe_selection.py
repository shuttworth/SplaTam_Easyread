"""
Code for Keyframe Selection based on re-projection of points from 
the current frame to the keyframes.
"""

import torch
import numpy as np


# 函数目的：从深度图和相机内参生成三维点云
def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    # A.内参解析
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    # B.计算采样像素的位置
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    # C.点云初始化
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    # D.移除相机原点处的点
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts

# 函数目的：选择与当前相机观测重叠的关键帧,并返回一组重叠程度较高的关键帧
# 输入参数：当前帧的真值深度 gt_depth，世界到相机的转换矩阵 w2c，相机内参 intrinsics，已有的关键帧列表 keyframe_list，需要选择的关键帧数量 k，以及可选的像素采样数量 pixels
# 返回值：返回选定的关键帧列表 selected_keyframe_list
def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_depth (tensor): ground truth depth image of the current frame.
            w2c (tensor): world to camera matrix (4 x 4).
            keyframe_list (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 1600.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        # Radomly Sample Pixel Indices from valid depth pixels
        # A. 随机采样像素索引
        # 首先，从当前帧的有效深度像素中（深度大于零的像素）随机选择一定数量（pixels）的像素索引sampled_indices
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        sampled_indices = valid_depth_indices[indices]

        # Back Project the selected pixels to 3D Pointcloud
        # B. 反投影选定的像素到3D点云
        # 利用 get_pointcloud 函数，将选定的像素索引反投影到3D点云空间, 得到的 pts 包含了在3D相机坐标系中的稀疏采样点的坐标
        # 注意：此utils/keyframe_selection.py里的get_pointcloud()函数，非彼scripts/splatam.py里的get_pointcloud()函数，函数同名 性质类似，但传参不同 实现有区别
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        list_keyframe = []
        # C. 遍历并进行关键帧重叠度分析
        for keyframeid, keyframe in enumerate(keyframe_list):
            # Get the estimated world2cam of the keyframe
            est_w2c = keyframe['est_w2c']
            # Transform the 3D pointcloud to the keyframe's camera space
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]

            # Filter out the points that are outside the image
            # 过滤出投影点在关键帧图像范围外的点
            edge = 20
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            # 计算在图像范围内的点占总投影点的比例(重叠百分比) percent_inside
            percent_inside = mask.sum()/projected_pts.shape[0]
            # 将关键帧的id和重叠百分比加入 list_keyframe 列表
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # D. 排序
        # Sort the keyframes based on the percentage of points that are inside the image
        # 根据重叠百分比对关键帧进行排序，百分比越高的排在前面
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        # 从排序后的关键帧列表中选择百分比大于零的前 k 个关键帧，即选择重叠程度最高的前 k 个关键帧作为最终选定的关键帧列表
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])

        return selected_keyframe_list
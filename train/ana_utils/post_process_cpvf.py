# Implemenation of post-processing in paper "CPVF: vectorization of agricultural cultivation field parcels via a boundary–parcel multi-task learning network in ultra-high-resolution remote sensing images"'


import cv2
import numpy as np
import os
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from Seg_Edge_2_Instance import postprocess_watershed_from_mask_and_edge,colorize_instances
# ====================== 配置路径 ========================
region_path = "region.png"   # 区域分割结果 (二值图)
boundary_path = "boundary.png"  # 边缘检测结果 (二值图)
save_dir = "./uvm_outputs"

boundary_path = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss_FinalEP/test_hq_edge/2019_31TCJ_patch_16_12.png'
region_path = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss_FinalEP/test_hq_mask/2019_31TCJ_patch_16_12.png'
save_dir = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss_FinalEP/Result_PostProcess/CPVF'
os.makedirs(save_dir, exist_ok=True)

# ====================== Step 0: 加载图像 ========================
region = cv2.imread(region_path, 0)  # 灰度图
boundary = cv2.imread(boundary_path, 0)
region = (region > 127).astype(np.uint8)
boundary = (boundary > 127).astype(np.uint8)

# ====================== Step 1: region-boundary interaction ========================
# dilated_boundary = cv2.dilate(boundary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
# masked_region = np.where((region == 1) & (dilated_boundary == 0), 1, 0).astype(np.uint8)

kernel = np.eye(3, dtype=np.uint8)
dilated_edge = cv2.dilate(boundary, kernel, iterations=1)
interacted_mask = np.where((region == 1) & (dilated_edge == 1), 0, region).astype(np.uint8) * 255
# cv2.imwrite(str(dilated_edge), dilated_edge)
cv2.imwrite(os.path.join(save_dir, "step1_masked_region.png"), interacted_mask)

# ====================== Step 2: skeletonization and topology ========================
skeleton = skeletonize(dilated_edge).astype(np.uint8)

# 提取骨架关键点（端点 & 交叉点）
def get_keypoints(skel):
    endpoints = np.zeros_like(skel)
    junctions = np.zeros_like(skel)
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    h, w = skel.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if skel[i,j] == 1:
                patch = skel[i-1:i+2, j-1:j+2]
                count = np.sum(patch * (kernel != 10))
                if count == 1:
                    endpoints[i,j] = 1
                elif count >= 3:
                    junctions[i,j] = 1
    return endpoints, junctions

endpoints, junctions = get_keypoints(skeleton)
cv2.imwrite(os.path.join(save_dir, "step2_skeleton.png"), skeleton * 255)
cv2.imwrite(os.path.join(save_dir, "step2_endpoints.png"), endpoints * 255)
cv2.imwrite(os.path.join(save_dir, "step2_junctions.png"), junctions * 255)

# ====================== Step 2.5: Overlay Vis
if len(boundary.shape) == 2:
    edge_rgb = cv2.cvtColor(boundary.astype(np.uint8), cv2.COLOR_GRAY2BGR)
else:
    edge_rgb = boundary.copy()
overlay = edge_rgb.copy() * 255
# 画红色骨架
overlay[skeleton.astype(bool)] = [255, 0, 0]  # Red (BGR)

# 画黄色 junctions：注意黄色 = 红 + 绿 => (255, 255, 0)
overlay[junctions.astype(bool)] = [0, 255, 255]  # Yellow (BGR)
# 保存图像
# if save_path is not None:
cv2.imwrite(os.path.join(save_dir, "step2_overlay.png"), overlay)

# ====================== Step 3: dangling line extension ========================
def extend_dangling_lines(skel, endpoints, max_length=30):
    extended = skel.copy()
    ys, xs = np.where(endpoints == 1)
    for y, x in zip(ys, xs):
        for angle in range(0, 360, 10):
            theta = np.deg2rad(angle)
            for l in range(1, max_length):
                dx = int(round(l * np.cos(theta)))
                dy = int(round(l * np.sin(theta)))
                nx, ny = x + dx, y + dy
                if 0 <= nx < skel.shape[1] and 0 <= ny < skel.shape[0]:
                    if extended[ny, nx] == 1:
                        break
                    extended[ny, nx] = 1
                else:
                    break
    return extended
def extend_dangling_lines_with_check(skel, endpoints, max_length=30):
    extended = skel.copy()
    h, w = skel.shape
    ys, xs = np.where(endpoints == 1)

    for y, x in zip(ys, xs):
        best_hit = None
        best_path = []

        for angle in range(0, 360, 10):
            theta = np.deg2rad(angle)
            path = []
            for l in range(1, max_length + 1):
                dx = int(round(l * np.cos(theta)))
                dy = int(round(l * np.sin(theta)))
                nx, ny = x + dx, y + dy

                if 0 <= nx < w and 0 <= ny < h:
                    if skel[ny, nx] == 1:
                        best_hit = (nx, ny)
                        break  # 当前方向命中
                    path.append((ny, nx))
                else:
                    break

            # 如果本方向命中，并且当前路径最短（或第一个命中），保留
            if best_hit and (not best_path or len(path) < len(best_path)):
                best_path = path

        # 若找到一个方向命中，保留该方向路径
        if best_path:
            for ny, nx in best_path:
                extended[ny, nx] = 1
        # 若所有方向都未命中，则不保留任何延伸

    return extended
def extend_dangling_lines_best_direction(skel, endpoints, max_length=30, angle_step=10):
    extended = skel.copy()
    h, w = skel.shape
    ys, xs = np.where(endpoints == 1)

    for y, x in zip(ys, xs):
        success = False  # 标记是否已有成功方向
        for angle in range(0, 360, angle_step):
            theta = np.deg2rad(angle)
            path = []  # 当前方向下的尝试路径
            for l in range(1, max_length + 1):
                dx = int(round(l * np.cos(theta)))
                dy = int(round(l * np.sin(theta)))
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if extended[ny, nx] == 1:
                        success = True
                        break  # 与已有线段相连，认为成功
                    path.append((ny, nx))
                else:
                    break  # 越界也停止
            if success:
                for ny, nx in path:
                    extended[ny, nx] = 1
                break  # 成功延伸后，不再尝试其它方向
    return extended
fixed_skeleton = extend_dangling_lines_with_check(skeleton, endpoints)
cv2.imwrite(os.path.join(save_dir, "step3_fixed_skeleton.png"), fixed_skeleton * 255)

# ====================== Step 4: watershed for instance segmentation ========================
# 结合修复骨架和 masked region，作为地块边界
combined = np.where((interacted_mask == 255) | (fixed_skeleton == 1), 1, 0).astype(np.uint8)

# 连通域作为种子
distance = ndi.distance_transform_edt(interacted_mask)
local_max = cv2.dilate(distance.astype(np.uint8), np.ones((3,3)), iterations=1)
markers = label(local_max > np.percentile(local_max, 90))
labels_ws = watershed(-distance, markers, mask=combined)

fixed_skeleton[fixed_skeleton==1]=255
region[region==1]=255
region_final_mask , region_final_mask_color_result = postprocess_watershed_from_mask_and_edge(region, fixed_skeleton, debug=False)
interacted_mask_final_mask , interacted_mask_final_mask_color_result = postprocess_watershed_from_mask_and_edge(interacted_mask, fixed_skeleton, debug=False)
# 可视化
plt.imsave(os.path.join(save_dir, "step4_instance_result_watershedByRegion.png"), region_final_mask_color_result, cmap='nipy_spectral')
plt.imsave(os.path.join(save_dir, "step4_instance_result_watershedByInteractedMask.png"), interacted_mask_final_mask_color_result, cmap='nipy_spectral')

print("✅ UVM处理流程执行完成，结果已保存至：", save_dir)

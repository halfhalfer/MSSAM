import cv2
import numpy as np
from skimage import measure
import numpy as np
from skimage.io import imread
import os
import random
import tqdm

def read_image(path):
    img = imread(path)  # shape: H x W x C
    # if img.dtype != np.float32:
    #     img = img.astype(np.float32) / 255.0
    return img
def generate_instance_from_semantic_and_edge(semantic_seg, edge_map, class_id=1, min_area=100):
    # Step 1: 提取语义掩膜
    semantic_mask = (semantic_seg == class_id).astype(np.uint8)

    # Step 2: 膨胀边缘图
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edge_map.astype(np.uint8), kernel, iterations=1)

    # Step 3: 从语义掩膜中扣除边缘
    cut_mask = semantic_mask.copy()
    cut_mask[dilated_edges > 0] = 0

    # Step 4: 连通域标记
    labeled_mask = measure.label(cut_mask, connectivity=1)
    
    # Step 5: 面积过滤
    instance_mask = np.zeros_like(labeled_mask)
    region_id = 1
    for region in measure.regionprops(labeled_mask):
        if region.area >= min_area:
            for coord in region.coords:
                instance_mask[coord[0], coord[1]] = region_id
            region_id += 1

    return instance_mask
def watershed_instance_segmentation(semantic_seg, edge_map, class_id=1, min_area=100):
    # Step 1: 提取目标类语义掩膜
    semantic_mask = (semantic_seg == class_id).astype(np.uint8)

    # Step 2: 边缘膨胀以增强切割效果
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edge_map.astype(np.uint8), kernel, iterations=1)

    # Step 3: 从语义中扣除边缘，生成“未知”区域
    cut_mask = semantic_mask.copy()
    cut_mask[dilated_edges > 0] = 0

    # Step 4: 连通区域标记作为前景种子
    num_labels, markers = cv2.connectedComponents(cut_mask)

    # Step 5: 添加背景与边缘为0，生成分水岭输入
    markers = markers + 1  # 确保背景为1
    markers[semantic_mask == 0] = 0  # 非目标区域为0

    # Step 6: 准备三通道图用于分水岭算法（必须是uint8 3通道）
    semantic_rgb = np.stack([semantic_mask * 255] * 3, axis=-1).astype(np.uint8)

    # Step 7: 应用Watershed算法
    cv2.watershed(semantic_rgb, markers)

    # Step 8: 移除边界标记（-1）和小区域
    instance_mask = np.zeros_like(markers, dtype=np.int32)
    label_id = 1
    for region_label in np.unique(markers):
        if region_label <= 1:  # 跳过背景和边缘
            continue
        region = (markers == region_label)
        if np.sum(region) >= min_area:
            instance_mask[region] = label_id
            label_id += 1

    return instance_mask

def colorize_instances(instance_mask):
    h, w = instance_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    instance_ids = np.unique(instance_mask)
    for instance_id in instance_ids:
        if instance_id == 0:
            continue  # 跳过背景
        color = [random.randint(0, 255) for _ in range(3)]
        color_mask[instance_mask == instance_id] = color

    return color_mask

import cv2
import numpy as np

def apply_marker_watershed(seg_map: np.ndarray, edge_map: np.ndarray, debug=False):
    """
    使用基于标记的分水岭算法对语义分割图像进行后处理，生成封闭的田块区域。

    参数：
    - seg_map: 二值语义分割图像（0 背景，255 前景）
    - edge_map: 二值边缘图像（0 非边界，255 边界）
    - debug: 若为 True，返回中间结果调试图

    返回：
    - watershed_result: 二值图，封闭的田块区域
    - color_result: 彩色可视化图，每个田块染色
    """
    # Step 1: 拓扑地形（反向边缘图）
    topo_surface = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 5)
    topo_surface = cv2.normalize(topo_surface, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 2: marker（田中心）
    dist_transform = cv2.distanceTransform(seg_map, cv2.DIST_L2, 5)
    _, markers_bin = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    markers_bin = np.uint8(markers_bin)
    num_labels, markers = cv2.connectedComponents(markers_bin)

    # Step 3: watershed
    seg_rgb = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(seg_rgb, markers.copy())
    watershed_result = np.zeros_like(seg_map)
    watershed_result[markers_ws > 1] = 255

    # 可视化染色
    color_result = np.zeros_like(seg_rgb)
    colors = np.random.randint(0, 255, (num_labels, 3), dtype=np.uint8)
    for label in range(2, num_labels):
        color_result[markers_ws == label] = colors[label]

    if debug:
        return watershed_result, color_result, topo_surface, markers
    else:
        return watershed_result, color_result
# def postprocess_watershed_from_mask_and_edge(mask: np.ndarray, edge: np.ndarray, debug=False):
#     """
#     使用语义分割图和边缘图进行后处理，生成封闭田块区域（语义前景内的分区）。
    
#     参数：
#     - mask: np.uint8，语义图（0为背景，255为前景）
#     - edge: np.uint8，边缘图（0为非边缘，255为边缘）
#     - debug: 是否返回中间结果图像

#     返回：
#     - final_mask: np.uint8，语义前景内的闭合田块分割图（0或255）
#     - color_result: np.uint8，彩色可视化图（背景为黑）
#     """
#     # Step 1: 边缘距离图
#     distance_map = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)

#     # Step 2: 拓扑表面（用于 watershed）
#     topo_surface = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     # Step 3: Marker from distance peaks
#     dist_thresh = 0.1 * distance_map.max()
#     _, marker_bin = cv2.threshold(distance_map, dist_thresh, 255, cv2.THRESH_BINARY)
#     marker_bin = marker_bin.astype(np.uint8)
#     num_markers, markers = cv2.connectedComponents(marker_bin)

#     # Step 4: Watershed on mask
#     mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     markers_ws = cv2.watershed(mask_rgb, markers.copy())  # 修改 markers 会影响原始数据

#     # Step 5: 构建 final mask，仅保留语义图前景（值为255）区域
#     final_mask = np.zeros_like(mask, dtype=np.uint8)
#     final_mask[(markers_ws > 1) & (mask == 255)] = 255  # 限定在语义前景区域

#     # Step 6: 构建彩色可视化图，背景为黑色
#     color_result = np.zeros_like(mask_rgb)
#     colors = np.random.randint(0, 255, (num_markers + 1, 3), dtype=np.uint8)
#     for label in range(2, num_markers + 1):  # 过滤背景和边界
#         color_result[(markers_ws == label) & (mask == 255)] = colors[label]

#     if debug:
#         return final_mask, color_result, distance_map, topo_surface, marker_bin
#     else:
#         return final_mask, color_result
def postprocess_watershed_from_mask_and_edge(mask: np.ndarray, edge: np.ndarray, debug=False):
    """
    使用语义分割图和边缘图进行后处理，生成封闭田块区域（语义前景内的实例掩膜）。

    参数：
    - mask: np.uint8，语义图（0为背景，255为前景）
    - edge: np.uint8，边缘图（0为非边缘，255为边缘）
    - debug: 是否返回中间结果图像

    返回：
    - final_mask: np.uint16，每个地块有唯一值（0为背景）
    - color_result: np.uint8，彩色可视化图（背景为黑）
    """
    # Step 1: 边缘距离图
    distance_map = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)

    # Step 2: 构建拓扑图
    topo_surface = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 3: 使用峰值区域构造 markers
    dist_thresh = 0.1 * distance_map.max()
    _, marker_bin = cv2.threshold(distance_map, dist_thresh, 255, cv2.THRESH_BINARY)
    marker_bin = marker_bin.astype(np.uint8)
    num_markers, markers = cv2.connectedComponents(marker_bin)

    # Step 4: Watershed
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(mask_rgb, markers.copy())  # 会输出边界为 -1

    # Step 5: 构建最终实例掩膜，背景为 0，其余每块田一个唯一值
    final_mask = np.zeros_like(mask, dtype=np.uint16)
    valid_region = (markers_ws > 1) & (mask == 255)
    final_mask[valid_region] = markers_ws[valid_region].astype(np.uint16)

    # Step 6: 彩色可视化
    color_result = np.zeros_like(mask_rgb)
    max_label = markers_ws.max()
    colors = np.random.randint(0, 255, (max_label + 1, 3), dtype=np.uint8)
    for label in range(2, max_label + 1):  # 过滤边界(-1)、背景(0,1)
        color_result[(markers_ws == label) & (mask == 255)] = colors[label]

    if debug:
        return final_mask, color_result, distance_map, topo_surface, marker_bin, markers
    else:
        return final_mask, color_result
def generate_markers_from_disjoint_instances(rough_instance_mask: np.ndarray):
    """
    若存在同一个 ID 有多个断开的区域（断裂实例），需重新编号以确保 watershed 正确。
    
    返回：
    - num_markers: 标记数
    - markers: 分水岭种子图
    """
    markers = np.zeros_like(rough_instance_mask, dtype=np.int32)
    current_label = 1
    unique_ids = np.unique(rough_instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # 跳过背景

    for uid in unique_ids:
        region_mask = (rough_instance_mask == uid).astype(np.uint8)
        num_subregions, subregions = cv2.connectedComponents(region_mask)
        for sid in range(1, num_subregions):  # 忽略背景
            markers[subregions == sid] = current_label
            current_label += 1

    num_markers = current_label - 1
    return num_markers, markers
# def generate_combined_markers(rough_instance_mask: np.ndarray, edge: np.ndarray):
#     """
#     将实例分割掩膜与基于边缘的 distance transform 提取的种子合并，作为最终 markers。
    
#     参数：
#     - rough_instance_mask: np.uint16，背景为0的实例掩膜
#     - edge: np.uint8，边缘图（0为非边缘，255为边缘）

#     返回：
#     - num_markers: int，最终合并后的种子数量
#     - combined_markers: np.int32，用于分水岭的 marker 图
#     """
#     # Step 1: 实例掩膜生成种子
#     bin_instance = (rough_instance_mask > 0).astype(np.uint8)
#     _, instance_markers = cv2.connectedComponents(bin_instance)

#     # Step 2: 基于边缘的 distance transform 生成种子
#     distance_map = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
#     dist_thresh = 0.1 * distance_map.max()
#     _, marker_bin = cv2.threshold(distance_map, dist_thresh, 255, cv2.THRESH_BINARY)
#     marker_bin = marker_bin.astype(np.uint8)
#     _, edge_markers = cv2.connectedComponents(marker_bin)

#     # Step 3: 合并两种 marker，避免 label 冲突
#     combined_markers = np.zeros_like(edge_markers, dtype=np.int32)

#     # 将 instance markers 直接拷贝进来（label 保留）
#     combined_markers[instance_markers > 0] = instance_markers[instance_markers > 0]

#     # 找当前最大 label，edge marker 需要递增防止覆盖
#     current_max = combined_markers.max()
#     edge_mask = (edge_markers > 0) & (combined_markers == 0)
#     combined_markers[edge_mask] = edge_markers[edge_mask] + current_max

#     num_markers = combined_markers.max()
#     return num_markers, combined_markers
def generate_combined_markers(rough_instance_mask: np.ndarray, edge: np.ndarray):
    """
    主体以边缘提取的markers为主，补充实例中有而边缘没有的区域。
    
    参数：
    - rough_instance_mask: np.uint16，实例掩膜（背景为0）
    - edge: np.uint8，边缘图（0为非边缘，255为边缘）

    返回：
    - num_markers: int，最终marker总数
    - combined_markers: np.int32，供分水岭使用的 marker 图
    """
    # Step 1: 基于边缘图生成主 markers
    distance_map = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    dist_thresh = 0.1 * distance_map.max()
    _, marker_bin = cv2.threshold(distance_map, dist_thresh, 255, cv2.THRESH_BINARY)
    marker_bin = marker_bin.astype(np.uint8)
    _, edge_markers = cv2.connectedComponents(marker_bin)
    edge_markers = edge_markers.astype(np.int32)

    # Step 2: 初始化 combined_markers 为 edge_markers（主力）
    combined_markers = edge_markers.copy()
    current_max = combined_markers.max()

    # Step 3: 寻找实例中存在但 edge 没有标记的区域
    mask_missing_in_edge = (rough_instance_mask > 0) & (edge_markers == 0)
    missing_mask = np.zeros_like(rough_instance_mask, dtype=np.uint8)
    missing_mask[mask_missing_in_edge] = 1

    # Step 4: 对这些区域做 connected components
    num_instance_extra, instance_extra = cv2.connectedComponents(missing_mask)
    instance_extra = instance_extra.astype(np.int32)

    # Step 5: 添加到 combined_markers 中，label 加偏移避免冲突
    for i in range(1, num_instance_extra):  # 跳过背景
        combined_markers[instance_extra == i] = current_max + i

    num_markers = combined_markers.max()
    return num_markers, combined_markers
def generate_combined_markers_cleaned_instance(
    rough_instance_mask: np.ndarray,
    edge: np.ndarray
):
    """
    使用边缘图为主导，结合清洗后的实例掩膜进行种子补全。

    步骤：
    1. 基于边缘图生成主 markers；
    2. 将 rough_instance_mask 做 connectedComponents，去除碎片；
    3. 寻找在清洗后的实例中有，但 edge 标记中没有的区域；
    4. 将这些区域补充到 markers 中。

    返回：
    - num_markers: int，最终 marker 数量
    - combined_markers: np.int32，分水岭输入标记图
    """
    # Step 1: 基于边缘图 distance transform 生成主 markers
    distance_map = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    dist_thresh = 0.1 * distance_map.max()
    _, marker_bin = cv2.threshold(distance_map, dist_thresh, 255, cv2.THRESH_BINARY)
    marker_bin = marker_bin.astype(np.uint8)
    _, edge_markers = cv2.connectedComponents(marker_bin)
    edge_markers = edge_markers.astype(np.int32)

    # Step 2: 清洗实例掩膜（去除碎片）
    cleaned_instance_mask = np.zeros_like(rough_instance_mask, dtype=np.int32)
    num_instances, instance_labels = cv2.connectedComponents((rough_instance_mask > 0).astype(np.uint8))
    for i in range(1, num_instances):
        cleaned_instance_mask[instance_labels == i] = i
    kernel = np.ones((3, 3), np.uint8)
    eroded_instance = cv2.erode((rough_instance_mask > 0).astype(np.uint8), kernel, iterations=1)
    _, cleaned_instance_markers = cv2.connectedComponents(eroded_instance)
    # Step 3: 找出实例中有、但 edge 标记中没有的区域（以 cleaned mask 为准）
    instance_only_mask = (cleaned_instance_mask > 0) & (edge_markers == 0)
    temp_mask = np.zeros_like(rough_instance_mask, dtype=np.uint8)
    temp_mask[instance_only_mask] = 1

    # Step 4: 连通域提取用于补充
    num_extra, extra_labels = cv2.connectedComponents(temp_mask)
    extra_labels = extra_labels.astype(np.int32)

    # Step 5: 合并：以 edge_markers 为主，补充 instance 部分
    combined_markers = edge_markers.copy()
    current_max = combined_markers.max()
    for i in range(1, num_extra):
        combined_markers[extra_labels == i] = current_max + i

    num_markers = combined_markers.max()
    return num_markers, combined_markers
def postprocess_watershed_from_mask_and_instance(mask: np.ndarray, instance: np.ndarray, debug=False):
    """
    使用语义分割图和破碎的实例图进行后处理，生成封闭田块区域（语义前景内的实例掩膜）。

    参数：
    - mask: np.uint8，语义图（0为背景，255为前景）
    - instance: np.uint16，实例图（0为背景，其余为实例编号））
    - debug: 是否返回中间结果图像

    返回：
    - final_mask: np.uint16，每个地块有唯一值（0为背景）
    - color_result: np.uint8，彩色可视化图（背景为黑）
    """
    # 使用实例图构造markers
    bin_mask = (instance > 0).astype(np.uint8)
    num_markers, markers = cv2.connectedComponents(bin_mask)
    # markers = bin_mask
    # Step 4: Watershed
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(mask_rgb, markers.copy())  # 会输出边界为 -1

    # Step 5: 构建最终实例掩膜，背景为 0，其余每块田一个唯一值
    final_mask = np.zeros_like(mask, dtype=np.uint16)
    valid_region = (markers_ws > 1) & (mask == 255)
    final_mask[valid_region] = markers_ws[valid_region].astype(np.uint16)

    # Step 6: 彩色可视化
    color_result = np.zeros_like(mask_rgb)
    max_label = markers_ws.max()
    colors = np.random.randint(0, 255, (max_label + 1, 3), dtype=np.uint8)
    for label in range(2, max_label + 1):  # 过滤边界(-1)、背景(0,1)
        color_result[(markers_ws == label) & (mask == 255)] = colors[label]

    if debug:
        return final_mask, color_result, markers
    else:
        return final_mask, color_result

def postprocess_watershed_from_mask_edge_instance(mask: np.ndarray,edge: np.ndarray, instance: np.ndarray, debug=False):
    """
    使用语义分割图和破碎的实例图进行后处理，生成封闭田块区域（语义前景内的实例掩膜）。

    参数：
    - mask: np.uint8，语义图（0为背景，255为前景）
    - instance: np.uint16，实例图（0为背景，其余为实例编号））
    - debug: 是否返回中间结果图像

    返回：
    - final_mask: np.uint16，每个地块有唯一值（0为背景）
    - color_result: np.uint8，彩色可视化图（背景为黑）
    """

    # Step 4: Watershed
    num_markers, markers =generate_combined_markers_cleaned_instance(instance, edge)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(mask_rgb, markers.copy())  # 会输出边界为 -1

    # Step 5: 构建最终实例掩膜，背景为 0，其余每块田一个唯一值
    final_mask = np.zeros_like(mask, dtype=np.uint16)
    valid_region = (markers_ws > 1) & (mask == 255)
    final_mask[valid_region] = markers_ws[valid_region].astype(np.uint16)

    # Step 6: 彩色可视化
    color_result = np.zeros_like(mask_rgb)
    max_label = markers_ws.max()
    colors = np.random.randint(0, 255, (max_label + 1, 3), dtype=np.uint8)
    for label in range(2, max_label + 1):  # 过滤边界(-1)、背景(0,1)
        color_result[(markers_ws == label) & (mask == 255)] = colors[label]

    if debug:
        return final_mask, color_result, markers
    else:
        return final_mask, color_result

import numpy as np
import cv2
from typing import Tuple

def generate_clean_watershed_instances(
    semantic_mask: np.ndarray,         # np.uint8, 0=background, 255=foreground
    edge: np.ndarray,                  # np.uint8, 0=non-edge, 255=edge
    rough_instance_mask: np.ndarray,  # np.uint16, 0=background, >0=instances
    area_threshold: int = 100         # post-filtering small regions
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用边缘图为主、结合清洗后的实例 mask 作为补充，生成更准确的 watershed 实例地块掩膜。

    返回：
    - final_mask: np.uint16，每个地块有唯一值（0为背景）
    - color_result: np.uint8，可视化彩色图
    """

    H, W = semantic_mask.shape

    # Step 1: 基于边缘生成主 markers（distance transform + threshold）
    distance_map = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 5)
    dist_thresh = 0.1 * distance_map.max()
    _, marker_bin = cv2.threshold(distance_map, dist_thresh, 255, cv2.THRESH_BINARY)
    marker_bin = marker_bin.astype(np.uint8)
    _, edge_markers = cv2.connectedComponents(marker_bin)
    edge_markers = edge_markers.astype(np.int32)

    # Step 2: 对 rough instance 做腐蚀 + connectedComponents（去除碎片）
    kernel = np.ones((3, 3), np.uint8)
    eroded_instance = cv2.erode((rough_instance_mask > 0).astype(np.uint8), kernel, iterations=1)
    _, cleaned_instance_markers = cv2.connectedComponents(eroded_instance)

    # Step 3: 找出 instance 有但 edge_markers 没覆盖的区域
    instance_only_mask = (cleaned_instance_markers > 0) & (edge_markers == 0)
    temp_mask = np.zeros_like(semantic_mask, dtype=np.uint8)
    temp_mask[instance_only_mask] = 1
    _, extra_labels = cv2.connectedComponents(temp_mask)
    extra_labels = extra_labels.astype(np.int32)

    # Step 4: 合并 markers（以 edge 为主，instance 为补充）
    combined_markers = edge_markers.copy()
    current_max = combined_markers.max()
    for i in range(1, extra_labels.max() + 1):
        combined_markers[extra_labels == i] = current_max + i

    # Step 5: 分水岭分割
    semantic_rgb = cv2.cvtColor(semantic_mask, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(semantic_rgb, combined_markers.copy())  # -1 是边界

    # Step 6: 提取前景区域中的有效实例
    valid_region = (markers_ws > 1) & (semantic_mask == 255) & (distance_map > 0)
    final_mask = np.zeros_like(semantic_mask, dtype=np.uint16)
    final_mask[valid_region] = markers_ws[valid_region].astype(np.uint16)

    # Step 7: 去除面积太小的噪声区域
    num_labels, labels = cv2.connectedComponents((final_mask > 0).astype(np.uint8))
    clean_mask = np.zeros_like(final_mask)
    label_id = 1
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area >= area_threshold:
            clean_mask[labels == i] = label_id
            label_id += 1

    # Step 8: 可视化上色
    color_result = np.zeros((H, W, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, (label_id + 1, 3), dtype=np.uint8)
    for i in range(1, label_id):
        color_result[clean_mask == i] = colors[i]

    return final_mask, color_result,combined_markers





def process_batch_watershed(seg_dir, edge_dir, save_bin_dir, save_rgb_dir):
    """
    对文件夹中的语义图和边缘图进行批处理，保存处理后的田块图和可视化图。

    参数：
    - seg_dir: 存放语义分割图（.png）的路径
    - edge_dir: 存放边缘图（.png）的路径
    - save_bin_dir: 处理后的田块二值图保存路径
    - save_rgb_dir: 可视化彩色图保存路径
    """
    os.makedirs(save_bin_dir, exist_ok=True)
    os.makedirs(save_rgb_dir, exist_ok=True)

    for file_name in tqdm.tqdm(os.listdir(seg_dir)):
        if not file_name.endswith('.png'):
            continue
        
        seg_path = os.path.join(seg_dir, file_name)
        edge_path = os.path.join(edge_dir, file_name)

        seg_map = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        edge_map = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

        if seg_map is None or edge_map is None:
            print(f"[跳过] {file_name} 无法读取图像")
            continue

        result_bin, result_rgb = postprocess_watershed_from_mask_and_edge(seg_map, edge_map)

        cv2.imwrite(os.path.join(save_bin_dir, file_name), result_bin)
        cv2.imwrite(os.path.join(save_rgb_dir, file_name), result_rgb)

        print(f"[处理完成] {file_name}")

def process_batch_watershed_Seg_Instance(seg_dir, instance_dir, save_bin_dir, save_rgb_dir):
    """
    对文件夹中的语义图和边缘图进行批处理，保存处理后的田块图和可视化图。

    参数：
    - seg_dir: 存放语义分割图（.png）的路径
    - instance_dir: 存放实例图（.png）的路径
    - save_bin_dir: 处理后的田块二值图保存路径
    - save_rgb_dir: 可视化彩色图保存路径
    """
    os.makedirs(save_bin_dir, exist_ok=True)
    os.makedirs(save_rgb_dir, exist_ok=True)

    for file_name in tqdm.tqdm(os.listdir(seg_dir)):
        if not file_name.endswith('.png'):
            continue
        
        seg_path = os.path.join(seg_dir, file_name)
        instance_path = os.path.join(instance_dir, file_name)

        seg_map = read_image(seg_path)#cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        instance__map = read_image(instance_path)#cv2.imread(instance_path, cv2.IMREAD_GRAYSCALE)

        if seg_map is None or instance__map is None:
            print(f"[跳过] {file_name} 无法读取图像")
            continue

        result_bin, result_rgb = postprocess_watershed_from_mask_and_instance(seg_map, instance__map)

        cv2.imwrite(os.path.join(save_bin_dir, file_name), result_bin)
        cv2.imwrite(os.path.join(save_rgb_dir, file_name), result_rgb)

        print(f"[处理完成] {file_name}")
if __name__ == '__main__':
    semantic_seg = read_image('/mnt/disk3/har/Param/Cropland/Ablations_channelToken/Pre_Result/ChannelToken_PEV2_5_DecoderV3_2_FullSample_Continue_Train_v4_smallLR_EdgeRefineLoss/test_hq_mask/NL_4215_S2_10m_256.png')
    edge_map = read_image('/mnt/disk3/har/Param/Cropland/Ablations_channelToken/Pre_Result/ChannelToken_PEV2_5_DecoderV3_2_FullSample_Continue_Train_v4_smallLR_EdgeRefineLoss/test_hq_edge/NL_4215_S2_10m_256.png')
    # instance_map = read_image('/mnt/disk3/har/Param/Cropland/SAM_HQ_Cropland_MultiSpectral/CBAM_Instance_sam_b_debug_ep30/test_hq_instance/NL_4985_S2_10m_256.png')
    np.random.seed(41)

    mask = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/CBAM_ForReview/test_hq_mask'
    edge = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/CBAM_ForReview/test_hq_edge'
    out_bin = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/CBAM_ForReview/parcels/bin'
    out_color = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/CBAM_ForReview/parcels/color_result'


    mask = '/mnt/disk3/har/Param/Cropland/S4A/ForReview_Exp/re_run_other_model/Model_Param/MainRe/Seed_41/ChannelToken_PEV2_5_2_DecoderV3_2/Result/ChannelToken_PEV2_5_2_DecoderV3_2/test_hq_mask'
    edge = '/mnt/disk3/har/Param/Cropland/S4A/ForReview_Exp/re_run_other_model/Model_Param/MainRe/Seed_41/ChannelToken_PEV2_5_2_DecoderV3_2/Result/ChannelToken_PEV2_5_2_DecoderV3_2/test_hq_edge'
    out_bin = '/mnt/disk3/har/Param/Cropland/S4A/ForReview_Exp/re_run_other_model/Model_Param/MainRe/Seed_41/ChannelToken_PEV2_5_2_DecoderV3_2/Result/ChannelToken_PEV2_5_2_DecoderV3_2/Parcels/bin'
    out_color = '/mnt/disk3/har/Param/Cropland/S4A/ForReview_Exp/re_run_other_model/Model_Param/MainRe/Seed_41/ChannelToken_PEV2_5_2_DecoderV3_2/Result/ChannelToken_PEV2_5_2_DecoderV3_2/Parcels/color_result'
    process_batch_watershed(mask, edge, out_bin, out_color)

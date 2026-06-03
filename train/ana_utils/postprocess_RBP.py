# Implemetation of post-processing In Paper "RBP-MTL: Agricultural Parcel Vectorization via  Region-Boundary-Parcel Decoupled  Multitask Learning"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def morphological_interaction_processing(edge_path, mask_path,save_path):
    # 路径处理
    edge_path = Path(edge_path)
    mask_path = Path(mask_path)

    # 输出路径设置
    dilated_edge_path = save_path +'/'+ edge_path.stem + "_dilated.png"
    interacted_mask_path = save_path +'/'+ mask_path.stem + "_interacted.png"

    # Step 1: 读取边缘图像，灰度模式
    edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
    if edge is None:
        raise FileNotFoundError(f"Edge image not found: {edge_path}")
    _, binary_edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)

    # Step 1: Morphological dilation（3x3 identity kernel）
    kernel = np.eye(3, dtype=np.uint8)
    dilated_edge = cv2.dilate(binary_edge, kernel, iterations=1)
    cv2.imwrite(str(dilated_edge_path), dilated_edge)

    # Step 2: 读取地块掩膜
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Step 2: 转为0/1进行交互操作
    mask_bin = (binary_mask > 127).astype(np.uint8)
    edge_bin = (dilated_edge > 127).astype(np.uint8)

    # Step 2: 交互处理：将边界区域从地块中移除
    result_mask = np.where((mask_bin == 1) & (edge_bin == 1), 0, mask_bin).astype(np.uint8) * 255
    cv2.imwrite(str(interacted_mask_path), result_mask)

    # # 可视化展示处理效果
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Parcel Mask")
    # plt.imshow(binary_mask, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title("Dilated Boundary")
    # plt.imshow(dilated_edge, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title("After Boundary-Object Interaction")
    # plt.imshow(result_mask, cmap='gray')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    print(f"[✓] Dilated edge saved to: {dilated_edge_path}")
    print(f"[✓] Interacted mask saved to: {interacted_mask_path}")
from skimage import morphology, measure

def postprocess_vectorize(mask_path,save_path,min_area=100):
    mask_path = Path(mask_path)
    save_mask_path = save_path +'/'+ mask_path.stem + '_cleaned.png'
    save_vector_path = save_path +'/'+ mask_path.stem + '_vector.shp'

    # 1. 读取交互后的mask（灰度）
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot find: {mask_path}")
    
    binary_mask = (mask > 127).astype(np.uint8)

    # 2. 填洞
    filled_mask = morphology.remove_small_holes(binary_mask.astype(bool), area_threshold=64).astype(np.uint8)

    # 3. 移除小斑点（小面积噪声）
    cleaned_mask = morphology.remove_small_objects(filled_mask.astype(bool), min_size=min_area).astype(np.uint8)

    # 4. 保存清理后的 mask
    cv2.imwrite(str(save_mask_path), cleaned_mask * 255)

    # 5. 可视化
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.title("Original")
    # plt.imshow(binary_mask, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title("Hole Filled")
    # plt.imshow(filled_mask, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title("Cleaned + Smoothed")
    # plt.imshow(cleaned_mask, cmap='gray')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    print(f"[✓] Cleaned binary mask saved to: {save_mask_path}")

    # # 6. 可选：导出为矢量图层（Shapefile）
    # if save_shapefile:
    #     contours = measure.find_contours(cleaned_mask, level=0.5)
    #     polygons = []
    #     for contour in contours:
    #         poly = Polygon(contour[:, ::-1])  # (row, col) -> (x, y)
    #         if poly.is_valid and poly.area > 10:
    #             polygons.append(poly)

    #     if polygons:
    #         gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")  # 若无地理信息，可用临时坐标系
    #         gdf.to_file(save_vector_path)
    #         print(f"[✓] Vectorized shapefile saved to: {save_vector_path}")
    #     else:
    #         print("[!] No valid polygons found for vectorization.")
import cv2
import numpy as np
from pathlib import Path
from skimage import morphology

def process_parcel_mask(edge_path, mask_path, save_path, min_area=100):
    edge_path = Path(edge_path)
    mask_path = Path(mask_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # 输出路径
    dilated_edge_path = save_path / (edge_path.stem + "_dilated.png")
    interacted_mask_path = save_path / (mask_path.stem + "_interacted.png")
    cleaned_mask_path = save_path / (mask_path.stem + "_cleaned.png")

    # Step 1: 读取边缘图像
    edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
    if edge is None:
        raise FileNotFoundError(f"Edge image not found: {edge_path}")
    _, binary_edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)

    # Step 1: Morphological dilation（3x3 identity kernel）
    kernel = np.eye(3, dtype=np.uint8)
    dilated_edge = cv2.dilate(binary_edge, kernel, iterations=1)
    cv2.imwrite(str(dilated_edge_path), dilated_edge)

    # Step 2: 读取掩膜图像
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Step 2: 边界交互处理
    mask_bin = (binary_mask > 127).astype(np.uint8)
    edge_bin = (dilated_edge > 127).astype(np.uint8)
    interacted_mask = np.where((mask_bin == 1) & (edge_bin == 1), 0, mask_bin).astype(np.uint8) * 255
    cv2.imwrite(str(interacted_mask_path), interacted_mask)

    # Step 3: 填洞
    filled_mask = morphology.remove_small_holes((interacted_mask > 127), area_threshold=64)
    
    # Step 4: 去除小区域噪声
    cleaned_mask = morphology.remove_small_objects(filled_mask, min_size=min_area).astype(np.uint8)

    # Step 5: 保存最终清理结果
    cv2.imwrite(str(cleaned_mask_path), cleaned_mask * 255)

    print(f"[✓] Dilated edge saved to: {dilated_edge_path}")
    print(f"[✓] Interacted mask saved to: {interacted_mask_path}")
    print(f"[✓] Cleaned mask saved to: {cleaned_mask_path}")

if __name__ == '__main__':
    # 实际调用
    edge_path = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss_FinalEP/test_hq_edge/2019_31TCJ_patch_16_12.png'
    mask_path = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss_FinalEP/test_hq_mask/2019_31TCJ_patch_16_12.png'
    save_path = '/mnt/disk3/har/Param/Cropland/S4A/SamHq/Result/ChannelToken_PEV2_5_2_DecoderV3_2_FinalEmbStructLoss_FinalEP/Result_PostProcess/RBP'
    process_parcel_mask(edge_path, mask_path, save_path, min_area=10)

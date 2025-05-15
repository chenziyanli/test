#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import traceback
import math
import sys
import os  # 导入 os 模块
import yaml  # 导入 yaml 模块

# --- 谱聚类需要的库 ---
try:
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import kneighbors_graph # 用于构建邻接图
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    rospy.logerr("SpectralDecomposer: scikit-learn not found! Install using: pip install -U scikit-learn")
    # 如果库不可用，直接退出可能比让节点崩溃更好
    sys.exit("Exiting due to missing scikit-learn dependency.")

class SpectralDecomposer:
    def __init__(self):
        rospy.init_node('spectral_area_decomposer', anonymous=True)
        rospy.loginfo("Spectral Area Decomposer Node Initializing...")

        # 地图相关属性
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None # geometry_msgs/Pose
        self.map_width = 0
        self.map_height = 0
        self.occupancy_grid = None # 2D numpy array
        self._map_received = False
        self._map_info = None # nav_msgs/MapMetaData

        # 可视化 Publisher
        self.roi_marker_pub = rospy.Publisher('~spectral_decomposition_rois', MarkerArray, queue_size=1, latch=True)

        # 订阅地图话题
        map_topic = rospy.get_param("~map_topic", "/map")
        rospy.loginfo(f"Subscribing to map topic: {map_topic}")
        # 使用回调函数处理地图消息
        self.map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self._map_callback, queue_size=1)

        rospy.loginfo("Spectral Decomposer initialized. Waiting for map...")

    def _map_callback(self, msg):
        """处理接收到的地图消息"""
        if self._map_received:
            # rospy.logdebug("Map already received.") # 可以取消注释用于调试
            return # 对于静态地图，通常只处理一次

        rospy.loginfo("Map received!")
        if msg.info.resolution <= 0:
            rospy.logerr("Received map with zero or negative resolution. Cannot process.")
            return
        if msg.info.width == 0 or msg.info.height == 0:
             rospy.logerr("Received map with zero width or height.")
             return

        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self._map_info = msg.info

        try:
            self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))
            self.occupancy_grid = self.map_data.astype(np.int8)
            self._map_received = True
            rospy.loginfo(f"Map processed. Shape: {self.occupancy_grid.shape}, Resolution: {self.map_resolution:.4f}")
            # 可选：处理完后取消订阅，如果只需要静态地图
            # self.map_sub.unregister()
            # rospy.loginfo("Unsubscribed from map topic after processing.")
        except Exception as e:
            rospy.logerr(f"Error processing map data: {e}")
            rospy.logerr(traceback.format_exc())
            self._map_received = False

    def is_ready(self):
        return self._map_received

    def get_free_space_cells(self, free_threshold=10):
        """获取空闲单元格坐标 (row, col) 列表及其索引映射"""
        if not self.is_ready():
            rospy.logwarn("Map not ready, cannot get free space cells.")
            return None, None
        # 阈值可以作为参数调整
        free_indices = np.where((self.occupancy_grid >= 0) & (self.occupancy_grid < free_threshold))
        if free_indices[0].size == 0:
            rospy.logwarn(f"No free space cells found with threshold < {free_threshold}.")
            return None, None
        free_cells_rc = list(zip(free_indices[0], free_indices[1]))
        cell_to_index_map = {cell: i for i, cell in enumerate(free_cells_rc)}
        rospy.loginfo(f"Found {len(free_cells_rc)} free space cells.")
        return free_cells_rc, cell_to_index_map

    def grid_to_world(self, row, col):
        """栅格坐标 (r, c) 到世界坐标 (x, y) - 中心点"""
        if not self.is_ready(): return None, None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        # 确保使用浮点数进行计算
        wx = float(ox) + (float(col) + 0.5) * float(self.map_resolution)
        wy = float(oy) + (float(row) + 0.5) * float(self.map_resolution)
        return wx, wy

    def world_to_grid(self, wx, wy):
        """世界坐标 (x, y) 到栅格坐标 (r, c)"""
        if not self.is_ready() or self.map_resolution == 0: return None, None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        # 确保使用浮点数进行计算
        col = int((float(wx) - float(ox)) / float(self.map_resolution))
        row = int((float(wy) - float(oy)) / float(self.map_resolution))
        if 0 <= row < self.map_height and 0 <= col < self.map_width:
            return row, col
        return None, None

    # ==========================================================================
    # === 修改后的 decompose_spectral 方法 ===
    # ==========================================================================
    def decompose_spectral(self, num_regions, free_threshold=10, n_neighbors=8):
        """使用谱聚类将空闲空间分割，并计算每个ROI的中心世界坐标"""
        rospy.loginfo(f"Attempting Spectral Clustering Decomposition into {num_regions} regions...")
        if not self.is_ready():
            rospy.logerr("Map not ready for decomposition.")
            return []
        if not _SKLEARN_AVAILABLE:
            rospy.logerr("Scikit-learn library not available for Spectral Clustering.")
            return []

        free_cells_rc, _ = self.get_free_space_cells(free_threshold)
        if not free_cells_rc:
            rospy.logerr("No free space found.")
            return []

        num_free_cells = len(free_cells_rc)
        if num_regions <= 0 or num_regions > num_free_cells:
             rospy.logerr(f"Invalid number of regions: {num_regions}.")
             return []

        rospy.loginfo("Building connectivity graph...")
        try:
            connectivity_matrix = kneighbors_graph(
                free_cells_rc,
                n_neighbors=n_neighbors,
                mode='connectivity',
                include_self=False,
                n_jobs=-1
            )
            connectivity_matrix = 0.5 * (connectivity_matrix + connectivity_matrix.T)
            rospy.loginfo("Connectivity graph built.")
        except Exception as e:
            rospy.logerr(f"Error building connectivity graph: {e}")
            return []

        rospy.loginfo(f"Performing Spectral Clustering (k={num_regions})...")
        try:
            sc = SpectralClustering(
                n_clusters=num_regions,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42,
                n_init=10,
                n_jobs=-1
            )
            labels = sc.fit_predict(connectivity_matrix)
            rospy.loginfo("Spectral Clustering finished.")
        except Exception as e:
            rospy.logerr(f"Error during Spectral Clustering fitting: {e}")
            return []

        rois = []
        for k in range(num_regions):
            cluster_indices = np.where(labels == k)[0]
            if cluster_indices.size == 0: continue

            cluster_cells = [free_cells_rc[i] for i in cluster_indices]

            # --- 新增：计算中心世界坐标 ---
            centroid_wx_sum = 0.0
            centroid_wy_sum = 0.0
            valid_cell_count = 0
            for r, c in cluster_cells:
                wx, wy = self.grid_to_world(r, c)
                if wx is not None and wy is not None:
                    centroid_wx_sum += wx
                    centroid_wy_sum += wy
                    valid_cell_count += 1

            if valid_cell_count > 0:
                centroid_world = (
                    float(centroid_wx_sum / valid_cell_count), # 确保是 float
                    float(centroid_wy_sum / valid_cell_count)  # 确保是 float
                )
            else:
                centroid_world = (0.0, 0.0) # 默认值
                rospy.logwarn(f"ROI {k} has no valid cells for centroid calculation.")
            # --- 结束新增 ---

            rows = [cell[0] for cell in cluster_cells]
            cols = [cell[1] for cell in cluster_cells]
            grid_bbox = {
                'min_r': int(min(rows)),
                'max_r': int(max(rows)),
                'min_c': int(min(cols)),
                'max_c': int(max(cols))
            }

            world_bbox = self.grid_roi_to_world_roi(grid_bbox)
            if world_bbox:
                world_bbox = {
                    'min_x': float(world_bbox['min_x']),
                    'min_y': float(world_bbox['min_y']),
                    'max_x': float(world_bbox['max_x']),
                    'max_y': float(world_bbox['max_y'])
                }
            else:
                world_bbox = {'min_x': 0.0, 'min_y': 0.0, 'max_x': 0.0, 'max_y': 0.0}
                rospy.logwarn(f"Could not calculate world_bbox for ROI {k}.")

            roi = {
                'id': k,
                'cells': [[int(r), int(c)] for r, c in cluster_cells], # 使用 Python int
                'grid_bbox': grid_bbox,
                'world_bbox': world_bbox,
                'centroid_world': centroid_world # <--- 添加中心点
            }
            rois.append(roi)
            # 修改日志输出以包含中心点
            rospy.loginfo(f"Generated ROI {k}: {len(cluster_cells)} cells, CentroidWorld=({centroid_world[0]:.3f}, {centroid_world[1]:.3f})")

        return rois
    # ==========================================================================
    # === 结束修改 ===
    # ==========================================================================

    def grid_roi_to_world_roi(self, grid_bbox):
        """将栅格坐标边界框转换为世界坐标边界框"""
        if not self.is_ready(): return None
        min_r, max_r = int(grid_bbox['min_r']), int(grid_bbox['max_r'])
        min_c, max_c = int(grid_bbox['min_c']), int(grid_bbox['max_c'])

        origin_x = float(self.map_origin.position.x)
        origin_y = float(self.map_origin.position.y)
        resolution = float(self.map_resolution)

        min_wx = origin_x + float(min_c) * resolution
        min_wy = origin_y + float(min_r) * resolution
        # 注意：最大坐标对应的是下一个单元格的起始位置
        max_wx = origin_x + float(max_c + 1) * resolution
        max_wy = origin_y + float(max_r + 1) * resolution

        return {
            'min_x': float(min_wx),
            'min_y': float(min_wy),
            'max_x': float(max_wx),
            'max_y': float(max_wy)
        }

    def visualize_rois(self, roi_list, duration=60.0):
        """在 RViz 中使用 Marker.CUBE_LIST 可视化分割后的空闲单元格"""
        if not roi_list:
            rospy.logwarn("No ROIs provided for visualization.")
            return
        if not self.is_ready():
             rospy.logwarn("Map not ready, cannot visualize ROIs.")
             return

        marker_array = MarkerArray()
        # 清除之前的标记
        delete_marker = Marker()
        delete_marker.header.frame_id = "map" # 使用地图坐标系
        delete_marker.ns = "roi_cells" # 统一的命名空间
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        # --- 同时清除中心点标记 ---
        delete_centroid_marker = Marker()
        delete_centroid_marker.header.frame_id = "map"
        delete_centroid_marker.ns = "roi_centroids" # 中心点使用不同命名空间
        delete_centroid_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_centroid_marker)
        # --- 结束清除 ---
        self.roi_marker_pub.publish(marker_array)
        rospy.sleep(0.1) # 短暂等待确保清除命令被处理

        marker_array = MarkerArray() # 创建新的 MarkerArray
        num_rois = len(roi_list)

        for roi in roi_list:
            roi_id = roi.get('id', -1)
            if roi_id == -1: continue

            cells_to_visualize = roi.get('cells', [])
            centroid_world = roi.get('centroid_world', None) # 获取中心点

            # --- 可视化 ROI 单元格 (CUBE_LIST) ---
            if cells_to_visualize:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "roi_cells" # 单元格命名空间
                marker.id = roi_id
                marker.type = Marker.CUBE_LIST
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0

                marker.scale.x = self.map_resolution
                marker.scale.y = self.map_resolution
                marker.scale.z = self.map_resolution * 0.5

                hue = (roi_id * 360.0 / max(1, num_rois)) % 360 # 避免除零
                try:
                    from colorsys import hsv_to_rgb
                    rgb = hsv_to_rgb(hue/360.0, 0.9, 0.9)
                    marker.color.r = float(rgb[0])
                    marker.color.g = float(rgb[1])
                    marker.color.b = float(rgb[2])
                except ImportError:
                    marker.color.r = float((roi_id * 50) % 255) / 255.0
                    marker.color.g = float((roi_id * 90) % 255) / 255.0
                    marker.color.b = float((roi_id * 130) % 255) / 255.0
                marker.color.a = 0.7 # 可以稍微透明一点

                marker.lifetime = rospy.Duration(duration)

                for r, c in cells_to_visualize:
                    wx, wy = self.grid_to_world(r, c)
                    if wx is not None and wy is not None:
                        p = Point()
                        p.x = wx
                        p.y = wy
                        p.z = 0.0 # 绘制在地面上
                        marker.points.append(p)

                if marker.points:
                    marker_array.markers.append(marker)

            # --- 新增：可视化 ROI 中心点 (SPHERE) ---
            if centroid_world:
                centroid_marker = Marker()
                centroid_marker.header.frame_id = "map"
                centroid_marker.header.stamp = rospy.Time.now()
                centroid_marker.ns = "roi_centroids" # 中心点命名空间
                centroid_marker.id = roi_id # 使用相同的 ID
                centroid_marker.type = Marker.SPHERE
                centroid_marker.action = Marker.ADD

                centroid_marker.pose.position.x = centroid_world[0]
                centroid_marker.pose.position.y = centroid_world[1]
                centroid_marker.pose.position.z = 0.1 # 比单元格稍高一点
                centroid_marker.pose.orientation.w = 1.0

                # 中心点大小可以比单元格大一点，颜色设为白色或与ROI同色但更亮/暗
                centroid_marker.scale.x = self.map_resolution * 3
                centroid_marker.scale.y = self.map_resolution * 3
                centroid_marker.scale.z = self.map_resolution * 3

                # 使用白色表示中心点
                centroid_marker.color.r = 1.0
                centroid_marker.color.g = 1.0
                centroid_marker.color.b = 1.0
                centroid_marker.color.a = 1.0 # 不透明

                centroid_marker.lifetime = rospy.Duration(duration)
                marker_array.markers.append(centroid_marker)
            # --- 结束新增 ---
            # --- 新增：可视化 ROI ID (TEXT_VIEW_FACING) ---
            if centroid_world: # ID 文本也需要一个位置，通常放在中心点附近
                id_text_marker = Marker()
                id_text_marker.header.frame_id = "map"
                id_text_marker.header.stamp = rospy.Time.now()
                id_text_marker.ns = "roi_id_text" # ID 文本命名空间
                id_text_marker.id = roi_id # 使用相同 ID
                id_text_marker.type = Marker.TEXT_VIEW_FACING # 类型为文本
                id_text_marker.action = Marker.ADD

                # 设置文本位置，可以放在中心点球体的正上方
                id_text_marker.pose.position.x = centroid_world[0]
                id_text_marker.pose.position.y = centroid_world[1]
                id_text_marker.pose.position.z = 0.2 + centroid_marker.scale.z / 2.0 # 放在球体上方一点
                id_text_marker.pose.orientation.w = 1.0

                # 设置文本内容
                id_text_marker.text = str(roi_id)

                # 设置文本大小 (scale.z 控制高度)
                id_text_marker.scale.z = 0.2 # 例如 0.2 米高

                # 设置文本颜色 (例如黑色)
                id_text_marker.color.r = 0.0
                id_text_marker.color.g = 0.0
                id_text_marker.color.b = 0.0
                id_text_marker.color.a = 1.0 # 不透明

                id_text_marker.lifetime = rospy.Duration(duration)
                marker_array.markers.append(id_text_marker)
            # --- 结束新增 ---
        if marker_array.markers:
            rospy.loginfo(f"Publishing {len(marker_array.markers)} ROI markers (cells & centroids) to {self.roi_marker_pub.resolved_name}")
            self.roi_marker_pub.publish(marker_array)
        else:
            rospy.logwarn("No valid markers generated for ROIs.")


# --- 主执行逻辑 ---
if __name__ == '__main__':
    try:
        decomposer = SpectralDecomposer()

        # 等待地图加载完成
        rate = rospy.Rate(1)
        while not decomposer.is_ready() and not rospy.is_shutdown():
            rospy.loginfo_once("Waiting for map...")
            try: rate.sleep()
            except rospy.ROSInterruptException: sys.exit()

        if rospy.is_shutdown(): sys.exit()

        rospy.loginfo("Map ready.")

        # --- 加载或执行谱聚类分解 ---
        num_regions_param = rospy.get_param("~num_regions", 9)
        free_thresh_param = rospy.get_param("~free_threshold", 10)
        knn_param = rospy.get_param("~k_neighbors", 8)
        cache_file = rospy.get_param("~roi_cache_file", "spectral_rois.yaml")

        generated_rois = None

        # 1. 尝试从文件加载
        if os.path.exists(cache_file):
            rospy.loginfo(f"Loading ROIs from cache file: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    cached_data = yaml.safe_load(f)
                    # Basic validation (ensure it's a list and contains expected keys, including grid_bbox)
                    if (isinstance(cached_data, list) and
                            all(isinstance(roi, dict) and
                                all(key in roi for key in ['id', 'cells', 'centroid_world', 'grid_bbox'])
                                for roi in cached_data)):
                        generated_rois = cached_data
                        rospy.loginfo(f"Successfully loaded {len(generated_rois)} ROIs from cache.")
                    else:
                        rospy.logwarn("Cache file has invalid format or missing keys (e.g., grid_bbox). Recalculating...")
                        generated_rois = None
            except Exception as e:
                rospy.logerr(f"Error loading ROIs from cache: {e}. Recalculating...")
                generated_rois = None

        # 2. 如果加载失败或文件不存在，则重新计算
        if generated_rois is None:
            rospy.loginfo("Performing Spectral Decomposition.")
            # Ensure decompose_spectral method includes centroid_world and grid_bbox calculation
            generated_rois = decomposer.decompose_spectral(
                num_regions=num_regions_param,
                free_threshold=free_thresh_param,
                n_neighbors=knn_param
            )

            # 3. 计算完成后，保存到文件
            if generated_rois:
                rospy.loginfo(f"Saving ROIs to cache file: {cache_file}")
                try:
                    with open(cache_file, 'w') as f:
                        yaml.safe_dump(generated_rois, f, default_flow_style=None)
                    rospy.loginfo(f"Successfully saved {len(generated_rois)} ROIs to cache.")
                except Exception as e:
                    rospy.logerr(f"Error saving ROIs to cache: {e}")

        # --- 可视化和输出结果 ---
        if generated_rois:
            rospy.loginfo(f"--- Spectral Decomposition Result ({len(generated_rois)} ROIs) ---")
            decomposer.visualize_rois(generated_rois) # Keep visualization

            # --- 输出为 sample_roi_list 格式 (使用 BBox 生成列表推导式) ---
            output_log_str = "sample_roi_list = [\n"
            for roi in generated_rois:
                roi_id = roi.get('id', 'N/A')
                grid_bbox = roi.get('grid_bbox') # 获取栅格边界框
                world_bbox = roi.get('world_bbox') # 获取世界边界框
                centroid_val = roi.get('centroid_world', (None, None))

                # Format centroid
                centroid_str = f"({centroid_val[0]:.3f}, {centroid_val[1]:.3f})" if all(v is not None for v in centroid_val) else "(N/A)"
                # Format world_bbox
                world_bbox_str = f"({world_bbox['min_x']:.3f}, {world_bbox['min_y']:.3f}), ({world_bbox['max_x']:.3f}, {world_bbox['max_y']:.3f})" if world_bbox else "(N/A)"
                # Format cells using list comprehension string from grid_bbox
                if grid_bbox and all(k in grid_bbox for k in ['min_r', 'max_r', 'min_c', 'max_c']):
                    min_r, max_r = grid_bbox['min_r'], grid_bbox['max_r']
                    min_c, max_c = grid_bbox['min_c'], grid_bbox['max_c']
                    # Create the list comprehension string using f-string formatting
                    # Remember range's upper bound is exclusive, so use max_r+1 and max_c+1
                    cells_repr = f"'[(r,c) for r in range({min_r}, {max_r + 1}) for c in range({min_c}, {max_c + 1})]'"
                else:
                    cells_repr = "'N/A (Missing grid_bbox)'" # Fallback if bbox is missing

                # Add comment similar to the example
                comment = f"# 区域{roi_id} 中心 {centroid_str} 附近"

                # Build dictionary string for this ROI, ensuring cells_repr is treated as a string literal part
                output_log_str += f"    {{'id': {roi_id}, 'cells': {cells_repr}, 'centroid_world': {centroid_str}}}, {comment}\n"
                output_log_str += f"    #world_bbox: {world_bbox}\n"
            # Remove trailing comma and newline if list is not empty
            if generated_rois:
                 output_log_str = output_log_str.rstrip(', \n') + "\n"

            output_log_str += "]"

            # Print the formatted string using rospy.loginfo
            rospy.loginfo("--- Generated ROIs (sample_roi_list format using BBox) ---\n" + output_log_str)
            # --- 结束格式化输出 ---

        else:
             rospy.logerr("Spectral decomposition failed to generate ROIs.")

        rospy.loginfo("Spectral Decomposer finished. Spinning.")
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down.")
    except Exception as e:
        rospy.logerr("An unexpected error occurred in SpectralDecomposer main.")
        rospy.logerr(traceback.format_exc())
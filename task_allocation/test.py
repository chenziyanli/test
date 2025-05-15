#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData # 导入 MapMetaData
from geometry_msgs.msg import Point, Quaternion, Vector3, Pose # 导入 Pose
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import traceback
import math
import sys
import os
import yaml

from task_allocation.msg import ROI, ROIList

# --- 谱聚类库检查 (保持不变) ---
try:
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import kneighbors_graph
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    rospy.logerr("SpectralDecomposer: scikit-learn not found!")
    sys.exit("Exiting due to missing scikit-learn dependency.")

class SpectralDecomposer:
    def __init__(self):
        rospy.init_node('spectral_area_decomposer', anonymous=True)
        rospy.loginfo("Spectral Area Decomposer Node Initializing...")

        # --- 读取参数 ---
        # 从参数服务器获取参数，提供默认值
        self.cluster_count = rospy.get_param("~cluster_count", 5)  # 期望的区域数量
        self.free_threshold = rospy.get_param("~free_threshold", 0) # OccupancyGrid 中低于此值的单元格被视为空闲 (0-100)
        self.n_neighbors = rospy.get_param("~n_neighbors", 8)      # 构建邻接图时考虑的邻居数量
        # 可选: 地图话题名和发布话题名
        map_topic = rospy.get_param("~map_topic", "/map")
        roi_list_topic = rospy.get_param("~roi_list_topic", "spectral_area_decomposer/spectral_roi_list")
        marker_topic = rospy.get_param("~marker_topic", "spectral_area_decomposer/roi_markers")

        rospy.loginfo(f"Parameters: cluster_count={self.cluster_count}, free_threshold={self.free_threshold}, n_neighbors={self.n_neighbors}")

        # 地图相关属性 (保持不变)
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = 0
        self.map_height = 0
        self.occupancy_grid = None
        self._map_received = False
        self._map_info = None

        # Publishers (使用参数设置话题名)
        self.roi_list_pub = rospy.Publisher(roi_list_topic, ROIList, queue_size=10)
        self.roi_marker_pub = rospy.Publisher(marker_topic, MarkerArray, queue_size=10) # 用于 RViz 可视化

        # Subscriber (使用参数设置话题名)
        self.map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback, queue_size=1)

        rospy.loginfo("SpectralDecomposer initialized. Waiting for map...")

    def is_ready(self):
        """检查地图数据是否已接收并有效"""
        return (self._map_received and
                self.map_data is not None and
                self._map_info is not None and
                self.occupancy_grid is not None and
                self.map_width > 0 and
                self.map_height > 0)

    def get_free_space_cells(self, free_threshold=0):
        """获取空闲空间的栅格坐标 (r, c) 和对应的值"""
        if not self.is_ready():
            return [], []
        # 查找低于阈值的单元格索引
        free_indices = np.where((self.occupancy_grid >= 0) & (self.occupancy_grid <= free_threshold))
        # 将索引转换为 (row, col) 列表
        free_cells_rc = list(zip(free_indices[0], free_indices[1]))
        free_values = self.occupancy_grid[free_indices].tolist()
        rospy.loginfo(f"Found {len(free_cells_rc)} free cells with threshold <= {free_threshold}.")
        return free_cells_rc, free_values

    def grid_to_world(self, r, c):
        """将栅格坐标 (r, c) 转换为世界坐标 (x, y)"""
        if not self.is_ready() or r < 0 or r >= self.map_height or c < 0 or c >= self.map_width:
            return None, None
        x = self.map_origin.position.x + (c + 0.5) * self.map_resolution
        y = self.map_origin.position.y + (r + 0.5) * self.map_resolution
        return x, y

    def grid_roi_to_world_roi(self, grid_bbox):
        """将栅格坐标的 BBox 转换为世界坐标的 BBox"""
        if not grid_bbox: return None
        min_r, max_r = grid_bbox['min_r'], grid_bbox['max_r']
        min_c, max_c = grid_bbox['min_c'], grid_bbox['max_c']

        # 计算四个角点的世界坐标
        # 注意：栅格单元的中心是 (c+0.5, r+0.5)，但边界通常对应整数栅格线
        min_x, min_y = self.grid_to_world(min_r - 0.5, min_c - 0.5) # 使用边界
        max_x, max_y = self.grid_to_world(max_r + 0.5, max_c + 0.5) # 使用边界

        if min_x is None or min_y is None or max_x is None or max_y is None:
            rospy.logwarn(f"无法转换 grid_bbox {grid_bbox} 的所有角点到世界坐标")
            return None

        return {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}


    def map_callback(self, msg):
        # ... (map_callback 保持不变, 但它会调用 decompose_area) ...
        if self._map_received:
            return # Process only the first map

        try:
            rospy.loginfo("Map received!")
            self._map_received = True
            self.map_data = msg # 存储整个 OccupancyGrid 消息
            self._map_info = msg.info # 存储 MapMetaData
            self.map_resolution = msg.info.resolution
            self.map_origin = msg.info.origin # 存储 Pose 对象
            self.map_width = msg.info.width
            self.map_height = msg.info.height
            # 转换地图数据为 numpy 数组 (注意 ROS 地图数据通常是 row-major)
            self.occupancy_grid = np.array(msg.data).reshape((self.map_height, self.map_width))
            rospy.loginfo(f"Map properties: Resolution={self.map_resolution:.3f}, Size=({self.map_width}x{self.map_height}), Origin=({self.map_origin.position.x:.2f}, {self.map_origin.position.y:.2f})")

            # 触发分解
            self.decompose_area()

        except Exception as e:
            rospy.logerr(f"处理地图回调时出错: {e}")
            rospy.logerr(traceback.format_exc())


    def decompose_area(self):
        """协调区域分解的主要函数"""
        if not self.is_ready():
            rospy.logwarn("Map data not available yet for decomposition.")
            return

        rospy.loginfo("Starting area decomposition...")

        # --- 调用真实的谱聚类函数 ---
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # 使用 __init__ 中获取的参数
        generated_rois = self.decompose_spectral(
            num_regions=self.cluster_count,
            free_threshold=self.free_threshold,
            n_neighbors=self.n_neighbors
        )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # --- 后续处理保持不变 ---
        if generated_rois: # 检查列表是否非空
            rospy.loginfo(f"Decomposition successful. Generated {len(generated_rois)} ROIs.")
            # 发布 ROI 列表消息
            self.publish_rois(generated_rois)
            # 发布 RViz 可视化标记
            self.publish_roi_markers(generated_rois)

            # --- (可选) 打印生成的 ROI 详细信息 ---
            # ... (可以保留或注释掉你的日志格式化代码) ...

        else:
             rospy.logerr("Spectral decomposition failed to generate ROIs.")


    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # vvvvvvv   这里是你的新函数 decompose_spectral   vvvvvvvvvvvv
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    def decompose_spectral(self, num_regions, free_threshold=0, n_neighbors=8): # 使用 free_threshold=0 作为默认值
        """使用谱聚类将空闲空间分割，并计算每个ROI的中心世界坐标"""
        rospy.loginfo(f"Attempting Spectral Clustering Decomposition into {num_regions} regions...")
        if not self.is_ready():
            rospy.logerr("Map not ready for decomposition.")
            return []
        if not _SKLEARN_AVAILABLE:
            rospy.logerr("Scikit-learn library not available for Spectral Clustering.")
            return []

        # 使用类方法获取空闲单元格
        free_cells_rc, _ = self.get_free_space_cells(free_threshold)
        if not free_cells_rc: # 检查列表是否为空
            rospy.logerr("No free space found for clustering.")
            return []

        num_free_cells = len(free_cells_rc)
        if num_regions <= 0 or num_regions > num_free_cells:
             # 修正日志级别为 WARN 或 ERROR
             rospy.logerr(f"Invalid number of regions requested: {num_regions}. Must be between 1 and {num_free_cells}.")
             return []

        rospy.loginfo(f"Building connectivity graph for {num_free_cells} free cells...")
        try:
            # 确保 free_cells_rc 是 numpy array 或可接受的格式
            connectivity_matrix = kneighbors_graph(
                np.array(free_cells_rc), # 显式转换为 numpy array
                n_neighbors=n_neighbors,
                mode='connectivity',
                include_self=False,
                n_jobs=-1 # 使用所有 CPU 核心
            )
            # 使矩阵对称
            connectivity_matrix = 0.5 * (connectivity_matrix + connectivity_matrix.T)
            rospy.loginfo("Connectivity graph built successfully.")
        except Exception as e:
            rospy.logerr(f"Error building connectivity graph: {e}")
            rospy.logerr(traceback.format_exc()) # 打印详细的回溯信息
            return []

        rospy.loginfo(f"Performing Spectral Clustering (k={num_regions})...")
        try:
            sc = SpectralClustering(
                n_clusters=num_regions,
                affinity='precomputed', # 因为我们提供了邻接矩阵
                assign_labels='kmeans', # 或 'discretize'
                random_state=42,        # 为了结果可复现
                n_init=10,              # kmeans 的初始化次数
                n_jobs=-1               # 使用所有 CPU 核心
            )
            # 使用连接矩阵进行拟合和预测
            labels = sc.fit_predict(connectivity_matrix)
            rospy.loginfo("Spectral Clustering finished.")
        except Exception as e:
            rospy.logerr(f"Error during Spectral Clustering fitting: {e}")
            rospy.logerr(traceback.format_exc()) # 打印详细的回溯信息
            return []

        # --- 处理聚类结果 ---
        rois = []
        unique_labels = np.unique(labels) # 获取实际生成的标签
        rospy.loginfo(f"Found {len(unique_labels)} unique clusters (expected {num_regions}).")

        for k in unique_labels: # 遍历实际找到的标签
            cluster_indices = np.where(labels == k)[0]
            if cluster_indices.size == 0: continue # 跳过空聚类 (理论上不应发生)

            # 获取属于该聚类的栅格坐标列表
            cluster_cells = [free_cells_rc[i] for i in cluster_indices]
            num_cells = len(cluster_cells) # 计算单元格数量

            # --- 计算中心世界坐标 ---
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
                centroid_world_tuple = (
                    float(centroid_wx_sum / valid_cell_count), # 确保是 float
                    float(centroid_wy_sum / valid_cell_count)  # 确保是 float
                )
            else:
                centroid_world_tuple = (0.0, 0.0) # 默认值 (或者使用其他策略)
                rospy.logwarn(f"ROI {k} has no valid cells for world centroid calculation.")

            # --- 计算栅格和世界坐标的 Bounding Box ---
            if num_cells > 0:
                rows = [cell[0] for cell in cluster_cells]
                cols = [cell[1] for cell in cluster_cells]
                grid_bbox = {
                    'min_r': int(min(rows)),
                    'max_r': int(max(rows)),
                    'min_c': int(min(cols)),
                    'max_c': int(max(cols))
                }
                # 尝试计算世界坐标 BBox
                world_bbox_dict = self.grid_roi_to_world_roi(grid_bbox)
                if world_bbox_dict:
                     # 确保字典键存在且值为 float
                     world_bbox_dict = {key: float(value) for key, value in world_bbox_dict.items()}
                else:
                     world_bbox_dict = {'min_x': 0.0, 'min_y': 0.0, 'max_x': 0.0, 'max_y': 0.0}
                     rospy.logwarn(f"Could not calculate world_bbox for ROI {k}.")
            else:
                grid_bbox = {'min_r': -1, 'max_r': -1, 'min_c': -1, 'max_c': -1} # 无效 BBox
                world_bbox_dict = {'min_x': 0.0, 'min_y': 0.0, 'max_x': 0.0, 'max_y': 0.0}

            # --- 构建 ROI 字典 ---
            # 注意：这里返回的是包含计算结果的 Python 字典
            # 下游函数 (publish_rois, publish_roi_markers) 需要将这些数据转换为 ROS 消息格式
            roi_dict = {
                'id': int(k), # 确保 ID 是整数
                'cells': [[int(r), int(c)] for r, c in cluster_cells], # 栅格坐标列表
                'num_cells': num_cells, # 添加单元格数量
                'grid_bbox': grid_bbox, # 栅格 BBox 字典
                'world_bbox': world_bbox_dict, # 世界坐标 BBox 字典
                'centroid_world': centroid_world_tuple # 世界坐标质心 (x, y) tuple
            }
            rois.append(roi_dict)
            rospy.loginfo(f"Generated ROI {k}: {num_cells} cells, CentroidWorld=({centroid_world_tuple[0]:.3f}, {centroid_world_tuple[1]:.3f})")

        return rois # 返回包含 ROI 字典的列表

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # vvvvvvv   结束 decompose_spectral 函数   vvvvvvvvvvvvvvvvvvv
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


    def publish_rois(self, generated_rois):
        """将生成的 ROI 数据发布为 ROIList 消息"""
        if not self.roi_list_pub:
            rospy.logwarn("ROI list publisher not initialized.")
            return
        if not generated_rois:
             rospy.loginfo("No ROIs generated to publish.")
             return
        if not self.is_ready(): # 检查地图信息是否可用
             rospy.logwarn("Map info not available for publishing ROIList header.")
             return

        roi_list_msg = ROIList()
        # 设置消息头
        roi_list_msg.header.stamp = rospy.Time.now()
        roi_list_msg.header.frame_id = self.map_data.header.frame_id if self.map_data is not None else "map"
        # 添加地图元数据
        if self._map_info:
            roi_list_msg.map_info = self._map_info

        # 遍历 decompose_spectral 返回的字典列表
        for roi_data in generated_rois:
            roi_msg = ROI() # 创建单个 ROI 消息

            # 从字典填充消息字段
            roi_msg.id = roi_data.get('id', -1)
            roi_msg.num_cells = roi_data.get('num_cells', 0) # 使用 num_cells 字段

            # 将质心元组 (x, y) 转换为 Point 对象
            centroid_tuple = roi_data.get('centroid_world')
            if centroid_tuple and isinstance(centroid_tuple, (tuple, list)) and len(centroid_tuple) == 2:
                roi_msg.centroid_world = Point(x=centroid_tuple[0], y=centroid_tuple[1], z=0.0)
            else:
                roi_msg.centroid_world = Point() # 默认值

            # (可选) 如果 ROI.msg 中有其他字段 (如 bbox)，可以在这里添加转换逻辑
            # if 'world_bbox' in roi_data: ...

            roi_list_msg.rois.append(roi_msg)

        self.roi_list_pub.publish(roi_list_msg)
        rospy.loginfo(f"Published ROIList with {len(roi_list_msg.rois)} ROIs.")


    def publish_roi_markers(self, generated_rois):
        """发布用于 RViz 可视化的 MarkerArray"""
        if not self.roi_marker_pub:
            rospy.logwarn("ROI marker publisher not initialized.")
            return
        if not generated_rois:
            rospy.loginfo("No ROIs generated, skipping marker publishing.")
            return
        if not self.is_ready(): # 检查地图信息
             rospy.logwarn("Map info not available for publishing markers.")
             return

        marker_array = MarkerArray()

        # --- 发布 DeleteAll 标记 (保持不变，使用 is not None 检查) ---
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = self.map_data.header.frame_id if self.map_data is not None else "map"
        # 删除所有相关命名空间
        for ns in ["roi_centroids", "roi_texts", "roi_bbox"]:
            delete_marker.ns = ns
            marker_array.markers.append(delete_marker) # 先添加到数组中

        # 先发布一次删除标记数组
        if marker_array.markers:
             self.roi_marker_pub.publish(marker_array)
             marker_array = MarkerArray() # 清空数组，准备添加新标记

        # 遍历 decompose_spectral 返回的字典列表
        for roi_data in generated_rois:
            marker_id = roi_data.get('id', -1) # 使用 ROI ID 作为基础标记 ID

            # --- 质心标记 (Sphere) ---
            marker_centroid = Marker()
            marker_centroid.header.frame_id = self.map_data.header.frame_id if self.map_data is not None else "map"
            marker_centroid.header.stamp = rospy.Time.now()
            marker_centroid.ns = "roi_centroids"
            marker_centroid.id = marker_id
            marker_centroid.type = Marker.SPHERE
            marker_centroid.action = Marker.ADD

            # 将质心元组 (x, y) 转换为 Point 对象用于姿态
            centroid_tuple = roi_data.get('centroid_world')
            if centroid_tuple and isinstance(centroid_tuple, (tuple, list)) and len(centroid_tuple) == 2:
                 marker_centroid.pose.position = Point(x=centroid_tuple[0], y=centroid_tuple[1], z=0.0)
            else:
                 marker_centroid.pose.position = Point() # 默认值

            marker_centroid.pose.orientation.w = 1.0
            marker_centroid.scale = Vector3(x=self.map_resolution*3, y=self.map_resolution*3, z=self.map_resolution*3)
            marker_centroid.color.a = 0.8; marker_centroid.color.r = 1.0; marker_centroid.color.g = 0.0; marker_centroid.color.b = 0.0
            marker_centroid.lifetime = rospy.Duration()
            marker_array.markers.append(marker_centroid)

            # --- 文本标记 (ROI ID) ---
            marker_text = Marker()
            marker_text.header.frame_id = self.map_data.header.frame_id if self.map_data is not None else "map"
            marker_text.header.stamp = rospy.Time.now()
            marker_text.ns = "roi_texts"
            marker_text.id = marker_id
            marker_text.type = Marker.TEXT_VIEW_FACING
            marker_text.action = Marker.ADD

            # 文本位置与质心相同 (或略微偏移)
            if centroid_tuple and isinstance(centroid_tuple, (tuple, list)) and len(centroid_tuple) == 2:
                marker_text.pose.position = Point(x=centroid_tuple[0], y=centroid_tuple[1], z=self.map_resolution * 5) # 稍微抬高 Z
            else:
                marker_text.pose.position = Point(z=self.map_resolution * 5)

            marker_text.scale.z = self.map_resolution * 10 # 文本高度
            marker_text.color.a = 1.0; marker_text.color.r = 1.0; marker_text.color.g = 1.0; marker_text.color.b = 1.0
            marker_text.text = str(marker_id)
            marker_text.lifetime = rospy.Duration()
            marker_array.markers.append(marker_text)

            # --- 世界坐标边界框标记 (LINE_STRIP) ---
            world_bbox_dict = roi_data.get('world_bbox')
            # 检查字典是否有效
            if world_bbox_dict and all(k in world_bbox_dict for k in ['min_x', 'max_x', 'min_y', 'max_y']):
                min_x = world_bbox_dict['min_x']
                max_x = world_bbox_dict['max_x']
                min_y = world_bbox_dict['min_y']
                max_y = world_bbox_dict['max_y']

                # 创建构成边界框的 5 个点 (闭合回路)
                points = [
                    Point(x=min_x, y=min_y, z=0.0), # 左下
                    Point(x=max_x, y=min_y, z=0.0), # 右下
                    Point(x=max_x, y=max_y, z=0.0), # 右上
                    Point(x=min_x, y=max_y, z=0.0), # 左上
                    Point(x=min_x, y=min_y, z=0.0)  # 回到左下闭合
                ]

                marker_bbox = Marker()
                marker_bbox.header.frame_id = self.map_data.header.frame_id if self.map_data is not None else "map"
                marker_bbox.header.stamp = rospy.Time.now()
                marker_bbox.ns = "roi_bbox"
                marker_bbox.id = marker_id
                marker_bbox.type = Marker.LINE_STRIP
                marker_bbox.action = Marker.ADD
                marker_bbox.pose.orientation.w = 1.0
                marker_bbox.scale.x = self.map_resolution * 0.5 # 线宽
                marker_bbox.color.a = 0.6; marker_bbox.color.r = 0.0; marker_bbox.color.g = 1.0; marker_bbox.color.b = 0.0
                marker_bbox.points = points # 设置点列表
                marker_bbox.lifetime = rospy.Duration()
                marker_array.markers.append(marker_bbox)

        # 发布包含所有新标记的 MarkerArray
        if marker_array.markers:
            self.roi_marker_pub.publish(marker_array)
            # 修正日志信息，包含实际添加的标记数量
            non_delete_markers = [m for m in marker_array.markers if m.action == Marker.ADD]
            rospy.loginfo(f"Published {len(non_delete_markers)} ADD markers for {len(generated_rois)} ROIs.")


# --- Main execution ---
if __name__ == '__main__':
    try:
        decomposer = SpectralDecomposer()
        rospy.loginfo("Spectral Decomposer Node is running. Waiting for map topic...")
        rospy.spin() # 保持节点运行，等待回调

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down Spectral Decomposer.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred in Spectral Decomposer: {e}")
        rospy.logerr(traceback.format_exc())
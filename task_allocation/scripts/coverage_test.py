#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
import numpy as np
import traceback
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import Quaternion,PoseStamped
import tf.transformations
class CoveragePlanner:
    def __init__(self):
        # 这个类的实例只负责规划，不需要是完整的 ROS 节点，
        # 但它需要访问地图数据。我们可以通过订阅或直接传递地图数据。
        # 为了简单，我们让它订阅一次地图。
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = 0
        self.map_height = 0
        self.occupancy_grid = None
        self._map_received = False

        # 短暂订阅地图话题获取地图
        map_topic = "/map"
        try:
            rospy.loginfo("CoveragePlanner: Waiting for map topic...")
            map_msg = rospy.wait_for_message(map_topic, OccupancyGrid, timeout=rospy.Duration(15.0))
            self._process_map_data(map_msg)
            rospy.loginfo("CoveragePlanner: Map received and processed.")
        except rospy.ROSException as e:
            rospy.logerr(f"CoveragePlanner: Failed to receive map from {map_topic}: {e}")
            # 可以抛出异常或设置一个错误状态

    def _process_map_data(self, msg):
        """处理接收到的 OccupancyGrid 消息"""
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))
        self.occupancy_grid = self.map_data # Store the grid
        self._map_received = True
        rospy.loginfo_once(f"CoveragePlanner: Map processed. Shape: {self.occupancy_grid.shape}, Resolution: {self.map_resolution}")

    def is_ready(self):
        """检查规划器是否已准备好（收到地图）"""
        return self._map_received

    # --- 坐标转换函数 ---
    def world_to_grid(self, wx, wy):
        if not self.is_ready(): return None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        # 简化：未考虑地图旋转，实际应用中可能需要
        col = int((wx - ox) / self.map_resolution)
        row = int((wy - oy) / self.map_resolution)
        print(f"world_to_grid: wx={wx}, wy={wy}, ox={ox}, oy={oy}, col={col}, row={row}")
        if 0 <= row < self.map_height and 0 <= col < self.map_width:
            return row, col
        return None

    def grid_to_world(self, row, col):
        if not self.is_ready(): return None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        # 简化：未考虑地图旋转
        wx = ox + (col + 0.5) * self.map_resolution
        wy = oy + (row + 0.5) * self.map_resolution
        return wx, wy

    # --- 检查栅格是否在 ROI 内且空闲 ---
    def is_cell_free_in_roi(self, row, col, roi):
        """
        检查栅格是否在指定的 ROI 内且空闲

        Args:
            row (int): 栅格行
            col (int): 栅格列
            roi (dict): {'min_x': float, 'min_y': float, 'max_x': float, 'max_y': float}

        Returns:
            bool: True 如果在 ROI 内且空闲
        """
        if self.occupancy_grid is None: return False
        # 1. 检查栅格索引
        if not (0 <= row < self.map_height and 0 <= col < self.map_width): return False
        # 2. 检查是否空闲 (0)
        if self.occupancy_grid[row, col] != 0: return False
        # 3. 检查栅格中心是否在世界坐标 ROI 内
        wx, wy = self.grid_to_world(row, col)
        if wx is None or wy is None: return False # 转换失败
        if not (roi['min_x'] <= wx < roi['max_x'] and \
                roi['min_y'] <= wy < roi['max_y']):
             return False
        return True

    # --- 覆盖路径规划算法 ---
    def plan_boustrophedon_path(self, roi):
        """
        为指定的 ROI 规划 Boustrophedon 路径。

        Args:
            roi (dict): {'min_x': float, 'min_y': float, 'max_x': float, 'max_y': float}

        Returns:
            list: 世界坐标下的航点列表 [(x1, y1), (x2, y2), ...]，如果失败则为空列表。
        """
        if not self.is_ready():
            rospy.logerr("Planner not ready (no map). Cannot plan path.")
            return []

        rospy.loginfo(f"Planning Boustrophedon path for ROI: {roi}")
        waypoints_world = []
        waypoints_grid = []

        # 将 ROI 转换为栅格坐标范围
        min_r, min_c = self.world_to_grid(roi['min_x'], roi['min_y'])
        max_r_corner, max_c_corner = self.world_to_grid(roi['max_x'], roi['max_y']) # 获取右上角对应的栅格

        # 注意：world_to_grid 返回的是包含该点的栅格，所以最大行列需要小心处理边界
        # 安全起见，稍微调整范围，确保覆盖到边界
        if None in [min_r, min_c, max_r_corner, max_c_corner]:
            rospy.logerr("ROI seems outside map boundaries.")
            return []
        
        # 确定实际需要扫描的栅格范围
        r_start = min(min_r, max_r_corner)
        r_end = max(min_r, max_r_corner)
        c_start = min(min_c, max_c_corner)
        c_end = max(min_c, max_c_corner)

        rospy.loginfo(f"Planning for grid ROI: rows {r_start}-{r_end}, cols {c_start}-{c_end}")

        # --- 简化的 Boustrophedon 逻辑 (逐行扫描) ---
        # (这里的逻辑与上一个回复中的类似，需要根据你的具体需求完善，
        #  特别是处理区域内障碍物和保证连通性的部分)
        direction = 1 # 1 for left-to-right (- C to + C), -1 for right-to-left (+ C to - C)
        last_waypoint_cell = None # Track last added cell to potentially add transition points

        for r in range(r_start, r_end + 1):
            segment_start_c = -1
            current_c_range = range(c_start, c_end + 1) if direction == 1 else range(c_end, c_start - 1, -1)
            print(f"current_c_range: {current_c_range}")
            row_has_free_space = False # Flag if any free space found in this row

            for c in current_c_range:
                if self.is_cell_free_in_roi(r, c, roi):
                    row_has_free_space = True
                    if segment_start_c == -1:
                        # 新段开始
                        segment_start_c = c
                        # 如果这不是第一行/第一个段，并且方向改变了，可能需要添加一个过渡点
                        # (简化处理：直接添加段起点)
                        if last_waypoint_cell != (r, segment_start_c):
                            waypoints_grid.append((r, segment_start_c))
                            last_waypoint_cell = (r, segment_start_c)

                elif segment_start_c != -1: # 段结束 (遇到障碍或离开ROI或地图边界)
                    segment_end_c = c - direction # 段的最后一个有效格子
                    # 添加段的终点（只有当它和起点不同时）
                    if last_waypoint_cell != (r, segment_end_c):
                         waypoints_grid.append((r, segment_end_c))
                         last_waypoint_cell = (r, segment_end_c)
                    segment_start_c = -1 # 重置

            # 处理行尾仍在段内的情况
            if segment_start_c != -1:
                segment_end_c = c_end if direction == 1 else c_start
                if last_waypoint_cell != (r, segment_end_c):
                     waypoints_grid.append((r, segment_end_c))
                     last_waypoint_cell = (r, segment_end_c)

            # 切换方向（只有在当前行找到了可走空间后才切换，避免空行来回摆动）
            if row_has_free_space:
                direction *= -1
        # --- Boustrophedon 逻辑结束 ---

        # 转换回世界坐标
        for r, c in waypoints_grid:
            wx, wy = self.grid_to_world(r, c)
            if wx is not None:
                # 避免添加距离过近的重复点 (可选)
                if not waypoints_world or \
                   math.sqrt((wx - waypoints_world[-1][0])**2 + (wy - waypoints_world[-1][1])**2) > self.map_resolution * 0.5:
                    waypoints_world.append((wx, wy))

        rospy.loginfo(f"Generated {len(waypoints_world)} world waypoints for ROI.")
        return waypoints_world

# 这个脚本本身不直接运行，它提供 CoveragePlanner 类给其他节点使用
# 可以添加一个简单的测试 main block
if __name__ == '__main__':
    try:
        rospy.init_node('coverage_planner_tester', anonymous=True)
        planner = CoveragePlanner()
        pub_topic="/coverage_path"
        path_publishers = rospy.Publisher(pub_topic, Path, queue_size=1, latch=True)
        if planner.is_ready():
            roi_test = {'min_x': 0.0, 'min_y': -2.0, 'max_x': 2.0, 'max_y': 0.0}
            waypoints = planner.plan_boustrophedon_path(roi=roi_test)
            # *** 新增：创建并发布 Path 消息 ***
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = rospy.Time.now()
            pose_stamped = PoseStamped()
            rospy.loginfo("Test Waypoints:")
            for i, (wx, wy) in enumerate(waypoints):
                     pose_stamped = PoseStamped()
                     # Path 中的 PoseStamped 也有自己的 header，通常也设为当前时间或路径生成时间
                     pose_stamped.header.stamp = path_msg.header.stamp
                     pose_stamped.header.frame_id = "map"
                     pose_stamped.pose.position.x = wx
                     pose_stamped.pose.position.y = wy
                     # 可以为路径点设置朝向（例如指向下一个点），或者用默认值
                     if i + 1 < len(waypoints):
                         next_x, next_y = waypoints[i+1]
                         yaw = math.atan2(next_y - wy, next_x - wx)
                         q = tf.transformations.quaternion_from_euler(0, 0, yaw)
                         pose_stamped.pose.orientation = Quaternion(*q)
                     else:
                         pose_stamped.pose.orientation.w = 1.0 # 最后一个点
                     path_msg.poses.append(pose_stamped)  
            path_publishers.publish(path_msg)
            rospy.loginfo(f" Published coverage path ({len(path_msg.poses)} poses).") 
        rospy.spin()  
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Error in planner tester.")
        rospy.logerr(traceback.format_exc())
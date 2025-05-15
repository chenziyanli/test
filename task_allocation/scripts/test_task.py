#!/usr/bin/env python3
# -*- coding: utf-8 -*- # 添加编码声明
import rospy
import numpy as np
import math
import random
import traceback
import sys
from functools import partial
from collections import defaultdict
import time # 用于计时
import re # 引入正则表达式库用于解析 cells 字符串

# ROS Msgs/Srvs
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from std_msgs.msg import Header # 引入 Header

# --- 依赖库导入与检查 ---
try:
    from deap import base, creator, tools, algorithms
    _DEAP_AVAILABLE = True
except ImportError:
    _DEAP_AVAILABLE = False
    print("ERROR: DEAP library not found! Install using: pip install deap", file=sys.stderr)

try:
    from pathfinding.core.grid import Grid
    from pathfinding.finder.a_star import AStarFinder
    from pathfinding.core.diagonal_movement import DiagonalMovement
    _PATHFINDING_AVAILABLE = True
except ImportError:
    _PATHFINDING_AVAILABLE = False
    print("ERROR: python-pathfinding library not found! Install using: pip install python-pathfinding", file=sys.stderr)

try:
    from scipy.ndimage import binary_dilation, generate_binary_structure
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    print("Warning: SciPy library not found. C-Space calculation for A* will be unavailable.", file=sys.stderr)

# --- Matplotlib 不再是必需的 ---

# --- 辅助函数：C-Space 计算 ---
def calculate_cspace(occupancy_grid, map_resolution, robot_radius_m, obstacle_threshold=50):
    """(辅助函数) 计算配置空间"""
    if not _SCIPY_AVAILABLE or occupancy_grid is None or map_resolution <= 0:
        rospy.logwarn("Cannot calculate C-Space: SciPy not available, grid missing, or invalid resolution.")
        return None

    if robot_radius_m <= 0:
        rospy.logwarn("Robot radius is zero or negative, using original occupancy grid as C-Space.")
        return (occupancy_grid >= obstacle_threshold) | (occupancy_grid == -1)

    robot_radius_cells = int(math.ceil(robot_radius_m / map_resolution))
    rospy.loginfo(f"Calculating C-Space with radius {robot_radius_m}m ({robot_radius_cells} cells)...")

    obstacle_mask = (occupancy_grid >= obstacle_threshold) | (occupancy_grid == -1)
    struct = generate_binary_structure(2, 2) # 8-connectivity

    try:
        start_time = time.time()
        cspace_grid = binary_dilation(obstacle_mask, structure=struct, iterations=robot_radius_cells)
        end_time = time.time()
        rospy.loginfo(f"C-Space calculation took {end_time - start_time:.3f} seconds.")
        return cspace_grid # 返回布尔型 NumPy 数组 (True=Obstacle/Inflated, False=Free)
    except Exception as e:
        rospy.logerr(f"Error during C-Space calculation (binary_dilation): {e}")
        return None


# ==============================================================================
# Class: TaskAllocator (GA for mTSP using A*)
# ==============================================================================
class TaskAllocator:
    """使用遗传算法(DEAP)和A*(python-pathfinding)成本估算来求解mTSP任务分配"""
    def __init__(self, robot_names, map_info=None, occupancy_grid=None, cspace_grid=None):
        rospy.loginfo("Task Allocator (GA-mTSP-A*) Initializing...")
        if not _DEAP_AVAILABLE or not _PATHFINDING_AVAILABLE:
            raise ImportError("Missing required libraries: DEAP and/or python-pathfinding.")

        self.robot_names = list(robot_names)
        self.num_robots = len(robot_names)
        if self.num_robots == 0:
            raise ValueError("TaskAllocator requires at least one robot name.")

        # 地图信息
        self.map_info = map_info
        self.occupancy_grid = occupancy_grid
        self.cspace_grid = cspace_grid
        self.map_resolution = map_info.resolution if map_info else None
        self.map_origin = map_info.origin if map_info else None
        self.map_width = map_info.width if map_info else 0
        self.map_height = map_info.height if map_info else 0
        self._map_ready = self._validate_map_data()

        # DEAP 创建器
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # 预计算成本存储
        self.roi_map = {}
        self.coverage_costs = {}
        self.travel_costs = {}
        self.start_costs = {}
        self.roi_idx_to_id = {}
        self.num_valid_rois = 0

        rospy.loginfo(f"Allocator initialized for {self.num_robots} robots.")
        if not self._map_ready:
            rospy.logwarn("Map data not provided or invalid during initialization.")

    def set_map_data(self, map_info, occupancy_grid, cspace_grid=None):
        """允许外部设置或更新地图数据"""
        self.map_info = map_info
        self.occupancy_grid = occupancy_grid
        self.cspace_grid = cspace_grid
        self.map_resolution = map_info.resolution if map_info else None
        self.map_origin = map_info.origin if map_info else None
        self.map_width = map_info.width if map_info else 0
        self.map_height = map_info.height if map_info else 0
        self._map_ready = self._validate_map_data()
        if self._map_ready:
             rospy.loginfo("Allocator map data updated.")
        else:
             rospy.logwarn("Invalid map data provided to set_map_data.")


    def _validate_map_data(self):
        """检查地图数据是否有效"""
        if self.map_info is None or \
           self.occupancy_grid is None or \
           self.map_resolution is None or self.map_resolution <= 0 or \
           self.map_origin is None or \
           self.map_width <= 0 or self.map_height <= 0:
           rospy.logwarn("Map validation failed: Missing or invalid map info/data.")
           return False
        if self.occupancy_grid.shape != (self.map_height, self.map_width):
             rospy.logerr(f"Occupancy grid shape {self.occupancy_grid.shape} does not match map info ({self.map_height}, {self.map_width})")
             return False
        if self.cspace_grid is not None and self.cspace_grid.shape != (self.map_height, self.map_width):
             rospy.logerr(f"C-Space grid shape {self.cspace_grid.shape} does not match map info ({self.map_height}, {self.map_width})")
             return False
        rospy.loginfo("Map data validated successfully.")
        return True

    def is_ready(self):
        """检查分配器是否有有效地图"""
        return self._map_ready

    def grid_to_world(self, row, col):
        """栅格坐标 (row, col) 到世界坐标 (x, y)"""
        if not self.is_ready(): return None, None
        if not (0 <= row < self.map_height and 0 <= col < self.map_width):
            # rospy.logwarn_throttle(10, f"grid_to_world: Input grid coordinate ({row}, {col}) out of bounds.")
            return None, None
        try:
            ox = self.map_origin.position.x
            oy = self.map_origin.position.y
            wx = ox + (col + 0.5) * self.map_resolution
            wy = oy + (row + 0.5) * self.map_resolution
            return wx, wy
        except Exception as e:
            rospy.logerr(f"Error in grid_to_world({row}, {col}): {e}")
            return None, None


    def world_to_grid(self, wx, wy):
        """世界坐标 (x, y) 到栅格坐标 (row, col)"""
        if not self.is_ready() or self.map_resolution == 0: return None, None
        try:
            ox = self.map_origin.position.x
            oy = self.map_origin.position.y
            dx = wx - ox
            dy = wy - oy
            # 使用 floor 而不是 int 来确保负坐标的正确处理
            col = math.floor(dx / self.map_resolution)
            row = math.floor(dy / self.map_resolution)
            # 检查计算出的栅格坐标是否在地图范围内
            if 0 <= row < self.map_height and 0 <= col < self.map_width:
                return int(row), int(col) # 返回整数类型
            else:
                # rospy.logwarn_throttle(10, f"world_to_grid: World coordinate ({wx:.2f}, {wy:.2f}) maps to grid ({row}, {col}), which is out of bounds.")
                return None, None
        except Exception as e:
             rospy.logerr(f"Error in world_to_grid({wx}, {wy}): {e}")
             return None, None


    def _estimate_coverage_cost(self, roi, coverage_speed=0.1, coverage_width=0.2):
        """估算覆盖 ROI 的时间成本 - 更新以处理字符串列表推导式"""
        coverage_speed = rospy.get_param("~allocation_coverage_speed", coverage_speed)
        coverage_width = rospy.get_param("~allocation_coverage_width", coverage_width)

        if not roi or 'cells' not in roi or not roi['cells'] or coverage_speed <= 0 or coverage_width <= 0:
            return 0.0
        if self.map_resolution is None or self.map_resolution <= 0:
            rospy.logwarn(f"Cannot estimate coverage cost for ROI {roi.get('id', 'N/A')}: Invalid map resolution.")
            return 0.0

        # --- 解析 'cells' ---
        num_cells = 0
        cells_data = roi['cells']

        if isinstance(cells_data, (list, tuple)):
            num_cells = len(cells_data)
        elif isinstance(cells_data, str):
            match = re.match(r"\s*\[\s*\(r,\s*c\)\s+for\s+r\s+in\s+range\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+for\s+c\s+in\s+range\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]\s*", cells_data)
            if match:
                try:
                    r_start, r_end, c_start, c_end = map(int, match.groups())
                    if r_end >= r_start and c_end >= c_start:
                        num_cells = (r_end - r_start) * (c_end - c_start)
                    else:
                        rospy.logwarn(f"Invalid range in 'cells' string for ROI {roi.get('id', 'N/A')}: r({r_start},{r_end}), c({c_start},{c_end})")
                except ValueError:
                    rospy.logwarn(f"Could not convert range values to int for ROI {roi.get('id', 'N/A')}: {match.groups()}")
                except Exception as e:
                    rospy.logwarn(f"Error calculating num_cells from parsed range for ROI {roi.get('id', 'N/A')}: {e}")
            else:
                try:
                    import ast
                    cell_list = ast.literal_eval(cells_data)
                    if isinstance(cell_list, (list, tuple)):
                        num_cells = len(cell_list)
                    else:
                        rospy.logwarn(f"Parsed 'cells' string for ROI {roi.get('id', 'N/A')} is not list/tuple.")
                except: # Catch all exceptions from literal_eval
                    rospy.logwarn(f"Could not parse 'cells' string for ROI {roi.get('id', 'N/A')}. Assuming 0 cells.")
        else:
            rospy.logwarn(f"ROI {roi.get('id', 'N/A')} has 'cells' in unexpected format: {type(cells_data)}.")


        if num_cells <= 0:
            return 0.0

        # 计算面积和覆盖时间
        area = num_cells * (self.map_resolution ** 2)
        effective_coverage_rate = coverage_speed * coverage_width
        coverage_time = area / effective_coverage_rate if effective_coverage_rate > 1e-9 else float('inf')

        return coverage_time


    def _calculate_astar_path_cost(self, start_rc, end_rc, debug_source="Unknown"):
        """
        使用 A* 算法计算两栅格点间的路径长度成本 (返回距离，单位：米)
        添加 debug_source 用于日志记录
        """
        if not self.is_ready():
             rospy.logwarn(f"A* Cost Calc [{debug_source}]: Allocator not ready.")
             return float('inf')
        if start_rc is None or end_rc is None:
             return float('inf')
        if start_rc == end_rc:
             return 0.0

        # 选择地图数据 (优先 C-Space)
        grid_matrix = None
        using_cspace = False
        if self.cspace_grid is not None:
            # **** MODIFICATION: Map True (Obstacle) -> 0, False (Free) -> 1 ****
            grid_matrix = np.where(self.cspace_grid, 0, 1).astype(np.int32)
            using_cspace = True
            rospy.logdebug(f"A* Cost Calc [{debug_source}]: Using C-Space (Obstacle=0, Free=1)")
        elif self.occupancy_grid is not None:
            rospy.logwarn_throttle(10, f"A* Cost Calc [{debug_source}]: C-Space unavailable, falling back to occupancy grid.")
            obstacle_threshold = rospy.get_param("~allocation_obstacle_threshold", 50)
            # Use Occupancy Grid: >= threshold or -1 (Obstacle) -> 0, Other (Free) -> 1
            grid_matrix = np.where(
                (self.occupancy_grid >= obstacle_threshold) | (self.occupancy_grid == -1), 0, 1
            ).astype(np.int32)
            rospy.logdebug(f"A* Cost Calc [{debug_source}]: Using Occupancy Grid (Obstacle=0, Free=1)")
        else:
            rospy.logerr(f"A* Cost Calc [{debug_source}]: No grid data available.")
            return float('inf')

        # 创建 pathfinding Grid 对象
        grid = None
        start_node = None
        end_node = None
        grid_matrix_T = grid_matrix.T # 转置一次，供 Grid 和调试使用

        # --- **** Final Check of Matrix Value Before Grid Creation **** ---
        start_col_chk, start_row_chk = start_rc[1], start_rc[0]
        end_col_chk, end_row_chk = end_rc[1], end_rc[0]
        start_val_before = "OOB"; end_val_before = "OOB"
        start_type_before = "N/A"; end_type_before = "N/A"
        if 0 <= start_col_chk < grid_matrix_T.shape[0] and 0 <= start_row_chk < grid_matrix_T.shape[1]:
            start_val_before = grid_matrix_T[start_col_chk, start_row_chk]
            start_type_before = type(start_val_before)
        if 0 <= end_col_chk < grid_matrix_T.shape[0] and 0 <= end_row_chk < grid_matrix_T.shape[1]:
            end_val_before = grid_matrix_T[end_col_chk, end_row_chk]
            end_type_before = type(end_val_before)
        rospy.loginfo(f"DEBUG Pre-Grid Check [{debug_source}]: Start ({start_col_chk},{start_row_chk}) val={start_val_before} ({start_type_before}), End ({end_col_chk},{end_row_chk}) val={end_val_before} ({end_type_before})")
        # --- **** End Added Check **** ---

        try:
            grid = Grid(matrix=grid_matrix_T)
            start_node = grid.node(start_rc[1], start_rc[0]) # node(col, row)
            end_node = grid.node(end_rc[1], end_rc[0])   # node(col, row)
        except IndexError:
             rospy.logerr(f"A* Index Error [{debug_source}]: Failed to create nodes. Start ({start_rc[1]},{start_rc[0]}), End ({end_rc[1]},{end_rc[0]}). Matrix shape (transposed): {grid_matrix_T.shape}. Using CSpace: {using_cspace}")
             return float('inf')
        except Exception as e:
            rospy.logerr(f"A* Error [{debug_source}]: Failed to create grid/nodes from ({start_rc}) to ({end_rc}): {e}")
            rospy.logerr(traceback.format_exc())
            return float('inf')

        # --- **** 详细调试 Walkable 检查 (保留日志，但依赖 grid.walkable) **** ---
        start_col, start_row = start_rc[1], start_rc[0]
        end_col, end_row = end_rc[1], end_rc[0]

        # 检查起点
        start_walkable = False
        start_node_obj_repr = "N/A"
        start_matrix_val = "N/A"
        start_walkable_result = "N/A"
        start_node_walkable_attr = "N/A"
        try:
            if start_node:
                start_node_obj_repr = repr(start_node)
                start_node_walkable_attr = getattr(start_node, 'walkable', 'Attribute Missing')
                if 0 <= start_col < grid_matrix_T.shape[0] and 0 <= start_row < grid_matrix_T.shape[1]:
                    start_matrix_val = grid_matrix_T[start_col, start_row]
                else:
                    start_matrix_val = "Out of Bounds"
                start_walkable_result = grid.walkable(start_col, start_row)
                start_walkable = start_node and start_walkable_result
            else:
                start_node_obj_repr = "Node creation failed or returned None"
                start_walkable = False

            if not start_walkable:
                 rospy.logwarn(f"--- A* Debug Walkable Start [{debug_source}] ---")
                 rospy.logwarn(f"  Start Coord (row, col): {start_rc}")
                 rospy.logwarn(f"  Checking (col, row): ({start_col}, {start_row})")
                 rospy.logwarn(f"  grid_matrix_T shape: {grid_matrix_T.shape}")
                 rospy.logwarn(f"  Value in grid_matrix_T[{start_col}, {start_row}]: {start_matrix_val}")
                 rospy.logwarn(f"  grid.node({start_col}, {start_row}) returned: {start_node_obj_repr}")
                 rospy.logwarn(f"  Node walkable attribute: {start_node_walkable_attr}")
                 rospy.logwarn(f"  grid.walkable({start_col}, {start_row}) returned: {start_walkable_result}")
                 rospy.logwarn(f"  Final Walkable Check: {start_walkable}")
                 rospy.logwarn(f"----------------------------------------------")
                 return float('inf')

        except Exception as e_walk_start:
            rospy.logerr(f"A* EXCEPTION during start walkable check [{debug_source}] for ({start_col},{start_row}): {e_walk_start}")
            return float('inf')


        # 检查终点 (类似起点)
        end_walkable = False
        end_node_obj_repr = "N/A"
        end_matrix_val = "N/A"
        end_walkable_result = "N/A"
        end_node_walkable_attr = "N/A"
        try:
            if end_node:
                end_node_obj_repr = repr(end_node)
                end_node_walkable_attr = getattr(end_node, 'walkable', 'Attribute Missing')
                if 0 <= end_col < grid_matrix_T.shape[0] and 0 <= end_row < grid_matrix_T.shape[1]:
                    end_matrix_val = grid_matrix_T[end_col, end_row]
                else:
                    end_matrix_val = "Out of Bounds"
                end_walkable_result = grid.walkable(end_col, end_row)
                end_walkable = end_node and end_walkable_result
            else:
                 end_node_obj_repr = "Node creation failed or returned None"
                 end_walkable = False

            if not end_walkable:
                 rospy.logwarn(f"--- A* Debug Walkable End [{debug_source}] ---")
                 rospy.logwarn(f"  End Coord (row, col): {end_rc}")
                 rospy.logwarn(f"  Checking (col, row): ({end_col}, {end_row})")
                 rospy.logwarn(f"  grid_matrix_T shape: {grid_matrix_T.shape}")
                 rospy.logwarn(f"  Value in grid_matrix_T[{end_col}, {end_row}]: {end_matrix_val}")
                 rospy.logwarn(f"  grid.node({end_col}, {end_row}) returned: {end_node_obj_repr}")
                 rospy.logwarn(f"  Node walkable attribute: {end_node_walkable_attr}")
                 rospy.logwarn(f"  grid.walkable({end_col}, {end_row}) returned: {end_walkable_result}")
                 rospy.logwarn(f"  Final Walkable Check: {end_walkable}")
                 rospy.logwarn(f"--------------------------------------------")
                 return float('inf')

        except Exception as e_walk_end:
            rospy.logerr(f"A* EXCEPTION during end walkable check [{debug_source}] for ({end_col},{end_row}): {e_walk_end}")
            return float('inf')
        # --- **** 详细调试结束 **** ---


        # 运行 A* 寻路
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path_rc_tuples = None
        try:
            # **** REMOVED Check: Removed the isinstance check using grid.node_class ****

            # Check if nodes are None before passing (should be caught by walkable check, but as safety)
            if start_node is None or end_node is None:
                 rospy.logerr(f"A* Path Cost INF [{debug_source}]: Start or End node is None before find_path. Start: {start_node}, End: {end_node}")
                 return float('inf')

            path_rc_tuples, runs = finder.find_path(start_node, end_node, grid)
            if not path_rc_tuples:
                rospy.logwarn(f"A* Path Cost INF [{debug_source}]: No path found from {start_rc} to {end_rc}. Using CSpace: {using_cspace}")
                return float('inf') # 没有找到路径

            # 计算路径长度
            path_length_m = 0.0
            for i in range(len(path_rc_tuples) - 1):
                c1, r1 = path_rc_tuples[i]
                c2, r2 = path_rc_tuples[i+1]
                w1_x, w1_y = self.grid_to_world(r1, c1)
                w2_x, w2_y = self.grid_to_world(r2, c2)
                if w1_x is not None and w2_x is not None:
                    try: dist = math.dist((w1_x, w1_y), (w2_x, w2_y))
                    except AttributeError: dist = math.sqrt((w1_x - w2_x)**2 + (w1_y - w2_y)**2)
                    path_length_m += dist
                else:
                    rospy.logwarn(f"A* path cost [{debug_source}]: Failed grid_to_world conversion for segment ({r1},{c1})->({r2},{c2}).")
                    return float('inf') # 转换失败，成本无效
            return path_length_m if path_length_m > 1e-9 else 1e-9

        except Exception as e:
            rospy.logerr(f"A* Path Cost INF [{debug_source}]: Exception during A* find_path from {start_rc} to {end_rc}: {e}")
            rospy.logerr(traceback.format_exc())
            return float('inf')


    def _precompute_costs(self, roi_list, robot_start_poses_map):
        """预计算所有旅行成本(使用A*)和覆盖成本"""
        rospy.loginfo("Precomputing costs using A*...")
        if not self.is_ready():
             rospy.logerr("Precompute Error: Allocator not ready (no map).")
             return False
        if not roi_list:
             rospy.logwarn("Precompute Warning: ROI list is empty.")

        # --- 1. 验证 ROI 并获取代表点 (质心) ---
        valid_rois = []
        roi_points = {} # ROI ID -> centroid_world (x, y)
        self.roi_map = {} # ROI ID -> full ROI dict
        rospy.loginfo(f"Validating {len(roi_list)} input ROIs...")
        num_skipped_outside = 0
        num_skipped_obstacle = 0
        for roi in roi_list:
            roi_id = roi.get('id')
            centroid = roi.get('centroid_world')
            cells_data = roi.get('cells')

            if roi_id is None or centroid is None or not cells_data:
                 rospy.logwarn(f"ROI {roi_id if roi_id is not None else 'Unknown'} missing id, centroid, or cells. Skipping.")
                 continue

            centroid_rc = self.world_to_grid(centroid[0], centroid[1])
            if centroid_rc is None:
                num_skipped_outside += 1
                continue

            is_walkable = True
            row, col = centroid_rc
            if not (0 <= row < self.map_height and 0 <= col < self.map_width):
                num_skipped_outside += 1
                continue

            if self.cspace_grid is not None:
                if self.cspace_grid[row, col]: # True means obstacle
                    is_walkable = False
                    num_skipped_obstacle += 1
                    continue
            elif self.occupancy_grid is not None:
                 obstacle_threshold = rospy.get_param("~allocation_obstacle_threshold", 50)
                 if self.occupancy_grid[row, col] >= obstacle_threshold or self.occupancy_grid[row, col] == -1:
                      is_walkable = False
                      num_skipped_obstacle += 1
                      continue

            if is_walkable:
                valid_rois.append(roi)
                roi_points[roi_id] = centroid
                self.roi_map[roi_id] = roi
        rospy.loginfo(f"ROI Validation Summary: Skipped {num_skipped_outside} (outside map), Skipped {num_skipped_obstacle} (in obstacle).")


        if not valid_rois:
            rospy.logerr("Precompute Error: No valid (reachable) ROIs found after validation.")
            return False

        self.num_valid_rois = len(valid_rois)
        valid_roi_ids = sorted(self.roi_map.keys())
        self.roi_idx_to_id = {i: roi_id for i, roi_id in enumerate(valid_roi_ids)}
        rospy.loginfo(f"Found {self.num_valid_rois} valid and reachable ROIs: {valid_roi_ids}")


        # --- 2. 估算覆盖成本 ---
        self.coverage_costs = {}
        rospy.loginfo("Estimating coverage costs...")
        for roi_id in valid_roi_ids:
            cost = self._estimate_coverage_cost(self.roi_map[roi_id])
            self.coverage_costs[roi_id] = cost
        rospy.loginfo("Coverage costs estimated.")


        # --- 3. 预计算旅行成本 (使用 A*) ---
        self.travel_costs = defaultdict(dict)
        self.start_costs = defaultdict(dict)
        rospy.loginfo(f"Calculating A* travel costs ({self.num_robots} robots, {self.num_valid_rois} ROIs)...")
        start_time_astar = time.time()
        inf_start_costs = 0
        inf_inter_roi_costs = 0


        #   a) 从机器人起始位置到各 ROI 的成本
        rospy.loginfo(" Calculating robot start costs to ROIs...")
        for r_name in self.robot_names:
            start_pose_xy = robot_start_poses_map.get(r_name)
            if start_pose_xy is None:
                 rospy.logwarn(f"Start pose not provided for robot {r_name}. Using default (0,0).")
                 start_pose_xy = (0.0, 0.0)

            start_rc = self.world_to_grid(start_pose_xy[0], start_pose_xy[1])
            if start_rc is None:
                rospy.logwarn(f"Start pose {start_pose_xy} for robot {r_name} is outside map bounds. Assigning infinite cost.")
                for roi_id in valid_roi_ids: self.start_costs[r_name][roi_id] = float('inf')
                inf_start_costs += len(valid_roi_ids)
                continue

            count_start = 0
            for roi_id in valid_roi_ids:
                 roi_centroid_xy = roi_points[roi_id]
                 end_rc = self.world_to_grid(roi_centroid_xy[0], roi_centroid_xy[1])
                 debug_tag = f"StartCost:{r_name}->{roi_id}"
                 if end_rc is None:
                      cost = float('inf')
                      rospy.logwarn(f"[{debug_tag}] End ROI {roi_id} centroid maps to None grid coord.")
                 else:
                      cost = self._calculate_astar_path_cost(start_rc, end_rc, debug_source=debug_tag)

                 self.start_costs[r_name][roi_id] = cost
                 if cost == float('inf'):
                      inf_start_costs += 1

                 count_start += 1
        #   b) ROI 之间的旅行成本
        num_pairs_to_calc = self.num_valid_rois * (self.num_valid_rois - 1) // 2
        count_inter_roi = 0
        total_pairs = self.num_valid_rois * (self.num_valid_rois -1)
        for i, roi_id_i in enumerate(valid_roi_ids):
            start_rc_i = self.world_to_grid(roi_points[roi_id_i][0], roi_points[roi_id_i][1])
            if start_rc_i is None:
                 rospy.logerr(f"Error precomputing inter-ROI cost: ROI {roi_id_i} centroid failed world_to_grid.")
                 for k_idx in range(i, self.num_valid_rois):
                      roi_id_k = valid_roi_ids[k_idx]
                      self.travel_costs[roi_id_i][roi_id_k] = float('inf')
                      self.travel_costs[roi_id_k][roi_id_i] = float('inf')
                      if i != k_idx: inf_inter_roi_costs += 1
                 continue

            for j in range(i, self.num_valid_rois):
                roi_id_j = valid_roi_ids[j]
                end_rc_j = self.world_to_grid(roi_points[roi_id_j][0], roi_points[roi_id_j][1])
                debug_tag = f"InterROICost:{roi_id_i}<->{roi_id_j}"

                if end_rc_j is None:
                     rospy.logerr(f"Error precomputing inter-ROI cost: ROI {roi_id_j} centroid failed world_to_grid.")
                     cost = float('inf')
                elif i == j:
                    cost = 0.0
                else:
                    cost = self._calculate_astar_path_cost(start_rc_i, end_rc_j, debug_source=debug_tag)
                    count_inter_roi += 1

                self.travel_costs[roi_id_i][roi_id_j] = cost
                self.travel_costs[roi_id_j][roi_id_i] = cost
                if i != j and cost == float('inf'):
                    inf_inter_roi_costs += 1

        end_time_astar = time.time()
        rospy.loginfo(f"Finished precomputing all A* costs in {end_time_astar - start_time_astar:.3f} seconds.")
        rospy.logwarn(f"Precomputation Summary: Found {inf_start_costs} infinite start costs and {inf_inter_roi_costs} infinite inter-ROI costs (unique pairs).")
        if inf_start_costs > 0 or inf_inter_roi_costs > 0:
             rospy.logwarn("WARNING: Infinite travel costs detected during precomputation. This will likely lead to infinite makespan in GA.")

        return True


    def _decode_chromosome(self, individual):
        """解码 DEAP 个体为每个机器人的 ROI 索引序列列表"""
        tours = []
        current_tour = []
        separator_value_start = self.num_valid_rois
        for item in individual:
            if item >= separator_value_start:
                tours.append(current_tour)
                current_tour = []
            else:
                if not isinstance(item, int) or item < 0 or item >= self.num_valid_rois:
                     rospy.logerr(f"Decode Error: Invalid item '{item}' found in individual: {individual}")
                     continue
                current_tour.append(item)
        tours.append(current_tour)
        while len(tours) < self.num_robots: tours.append([])
        if len(tours) > self.num_robots:
             rospy.logerr(f"Decode Error: Decoded more tours ({len(tours)}) than robots ({self.num_robots}). Individual: {individual}")
             tours = tours[:self.num_robots]
        return tours


    def _evaluate_makespan(self, individual, robot_start_poses_map):
        """DEAP 适应度评估函数：计算 Makespan (最大完成时间)"""
        if not self.roi_map or not self.roi_idx_to_id or not self.start_costs or not self.travel_costs:
            rospy.logerr("Eval Error: Precomputation data missing.")
            return (float('inf'),)

        robot_roi_indices_tours = self._decode_chromosome(individual)
        if robot_roi_indices_tours is None or len(robot_roi_indices_tours) != self.num_robots:
             rospy.logerr(f"Eval Error: Decoding failed for individual: {individual}")
             return(float('inf'),)

        travel_speed = rospy.get_param("~allocation_travel_speed", 0.5)
        if travel_speed <= 1e-6:
             travel_speed = 1.0

        robot_completion_times = []
        infinite_cost_encountered = False

        for i, tour_indices in enumerate(robot_roi_indices_tours):
            if i >= len(self.robot_names): continue
            robot_name = self.robot_names[i]
            current_robot_time = 0.0
            last_roi_id = None

            if not tour_indices:
                 robot_completion_times.append(current_robot_time)
                 continue

            # --- Start to First ROI ---
            first_roi_idx = tour_indices[0]
            first_roi_id = self.roi_idx_to_id.get(first_roi_idx)
            if first_roi_id is None:
                 rospy.logerr(f"Eval Error: Invalid first ROI index {first_roi_idx} for {robot_name}. Indi: {individual}")
                 infinite_cost_encountered = True; break

            start_travel_dist = self.start_costs.get(robot_name, {}).get(first_roi_id, float('inf'))
            if start_travel_dist == float('inf'):
                 infinite_cost_encountered = True; break

            current_robot_time += start_travel_dist / travel_speed
            current_robot_time += self.coverage_costs.get(first_roi_id, 0.0)
            last_roi_id = first_roi_id

            # --- Subsequent ROIs ---
            for j in range(1, len(tour_indices)):
                current_roi_idx = tour_indices[j]
                current_roi_id = self.roi_idx_to_id.get(current_roi_idx)
                if current_roi_id is None:
                     rospy.logerr(f"Eval Error: Invalid ROI index {current_roi_idx} for {robot_name}. Indi: {individual}")
                     infinite_cost_encountered = True; break

                travel_dist = self.travel_costs.get(last_roi_id, {}).get(current_roi_id, float('inf'))
                if travel_dist == float('inf'):
                     infinite_cost_encountered = True; break

                current_robot_time += travel_dist / travel_speed
                current_robot_time += self.coverage_costs.get(current_roi_id, 0.0)
                last_roi_id = current_roi_id

            if infinite_cost_encountered: break
            robot_completion_times.append(current_robot_time)

        if infinite_cost_encountered:
            return (float('inf'),)

        makespan = max(robot_completion_times) if robot_completion_times else 0.0
        return (makespan,)


    def allocate_tasks(self, roi_list, robot_start_poses_map):
        """主分配函数，运行 GA 求解 mTSP"""
        if not self.is_ready():
             rospy.logerr("Allocation Error: Allocator not ready.")
             return {}

        rospy.loginfo("Starting Task Allocation using GA for mTSP (A* Costs)...")

        if not self._precompute_costs(roi_list, robot_start_poses_map):
            rospy.logerr("Task allocation failed: Cost precomputation error.")
            if self.num_valid_rois == 0:
                 rospy.logerr(" -> Reason: No valid ROIs found after validation.")
            return {}
        if self.num_valid_rois == 0:
             rospy.logwarn("No valid ROIs to allocate after precomputation. Returning empty assignment.")
             return {name: [] for name in self.robot_names}

        if self.num_valid_rois < self.num_robots:
            rospy.logwarn(f"Num ROIs ({self.num_valid_rois}) < Num Robots ({self.num_robots}). Some robots may get no tasks.")

        # 设置 DEAP 工具箱
        toolbox = base.Toolbox()
        num_separators = self.num_robots - 1
        elements = list(range(self.num_valid_rois)) + list(range(self.num_valid_rois, self.num_valid_rois + num_separators))
        toolbox.register("indices", random.sample, elements, len(elements))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_makespan, robot_start_poses_map=robot_start_poses_map)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 运行遗传算法
        population_size = rospy.get_param("~ga_population_size", 100)
        num_generations = rospy.get_param("~ga_num_generations", 80)
        crossover_prob = rospy.get_param("~ga_crossover_prob", 0.8)
        mutation_prob = rospy.get_param("~ga_mutation_prob", 0.2)

        rospy.loginfo(f"Running GA: Pop={population_size}, Gen={num_generations}, CXPB={crossover_prob}, MUTPB={mutation_prob}")
        rospy.loginfo(f"Problem Size: NumValidROIs={self.num_valid_rois}, NumRobots={self.num_robots}")
        start_ga_time = time.time()

        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('inf'))
        stats.register("avg", np.mean); stats.register("min", np.min); stats.register("max", np.max)

        try:
            algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                                ngen=num_generations, stats=stats, halloffame=hof, verbose=True)
        except Exception as e:
             rospy.logerr(f"Error during GA execution (eaSimple): {e}")
             rospy.logerr(traceback.format_exc())
             return {}

        end_ga_time = time.time()
        rospy.loginfo(f"GA execution finished in {end_ga_time - start_ga_time:.3f} seconds.")

        # 处理结果
        if not hof or not hof[0].fitness.valid:
             rospy.logerr("GA Hall of Fame is empty or contains invalid individual. No solution found.")
             return {}

        best_individual = hof[0]
        best_makespan = best_individual.fitness.values[0]

        if best_makespan == float('inf'):
            rospy.logerr("GA finished, but the best solution found still has infinite makespan.")
            return {}

        rospy.loginfo(f"GA Best Solution Found: Estimated Makespan = {best_makespan:.2f} seconds")

        best_tours_indices = self._decode_chromosome(best_individual)

        final_assignment = defaultdict(list)
        assigned_roi_ids = set()
        total_assigned_rois_count = 0

        for i, tour_indices in enumerate(best_tours_indices):
            if i >= len(self.robot_names): continue
            robot_name = self.robot_names[i]
            robot_tour_rois = []
            for roi_idx in tour_indices:
                roi_id = self.roi_idx_to_id.get(roi_idx)
                if roi_id is not None and roi_id in self.roi_map:
                    robot_tour_rois.append(self.roi_map[roi_id])
                    assigned_roi_ids.add(roi_id)
                    total_assigned_rois_count += 1
                else:
                     rospy.logwarn(f"Warning: Invalid ROI index/id {roi_idx}/{roi_id} in best tour for {robot_name}.")
            final_assignment[robot_name] = robot_tour_rois

        rospy.loginfo("--- Final Task Allocation (Robot -> Ordered ROI IDs) ---")
        unassigned_rois = set(self.roi_map.keys()) - assigned_roi_ids
        for robot, rois in final_assignment.items():
            roi_ids_ordered = [r['id'] for r in rois]
            rospy.loginfo(f"  {robot}: Assigned ROIs {roi_ids_ordered}")
        rospy.loginfo(f"Total ROIs assigned in best solution: {total_assigned_rois_count} / {self.num_valid_rois}")
        if unassigned_rois:
             rospy.logwarn(f"Warning: The following valid ROIs were not assigned: {sorted(list(unassigned_rois))}")
        rospy.loginfo("-----------------------------------------------------")

        return dict(final_assignment)


# ==============================================================================
# Main Execution Block (示例用法)
# ==============================================================================
if __name__ == '__main__':
    # 检查核心依赖
    if not _DEAP_AVAILABLE or not _PATHFINDING_AVAILABLE:
        rospy.logfatal("Missing required libraries (DEAP and/or python-pathfinding).")
        sys.exit(1)

    try:
        rospy.init_node('task_allocator_ga_mtsp_astar_node', anonymous=True)
        rospy.loginfo("Task Allocator Node Started.")

        # --- 1. 获取地图数据 ---
        map_msg = None
        map_topic = rospy.get_param("~map_topic", "/map")
        rospy.loginfo(f"Waiting for map on topic '{map_topic}'...")
        try:
            map_msg = rospy.wait_for_message(map_topic, OccupancyGrid, timeout=20.0)
            rospy.loginfo("Map received successfully.")
        except rospy.ROSException as e:
            rospy.logfatal(f"Failed to receive map from '{map_topic}': {e}. Exiting.")
            sys.exit(1)
        except Exception as e:
             rospy.logfatal(f"Error receiving map: {e}")
             sys.exit(1)

        map_info = map_msg.info
        occupancy_grid = np.array(map_msg.data, dtype=np.int8).reshape((map_info.height, map_info.width))
        rospy.loginfo(f"Map Info: Res={map_info.resolution:.3f}, Size=({map_info.width}x{map_info.height}), Origin=({map_info.origin.position.x:.2f}, {map_info.origin.position.y:.2f}), Frame='{map_msg.header.frame_id}'")


        # --- 2. 计算 C-Space ---
        robot_radius = rospy.get_param("~robot_radius", 0.3)
        obstacle_thresh = rospy.get_param("~obstacle_threshold", 65)
        rospy.loginfo(f"Calculating C-Space with radius {robot_radius:.2f}m and threshold {obstacle_thresh}...")
        cspace_grid = calculate_cspace(occupancy_grid, map_info.resolution, robot_radius, obstacle_thresh)

        if cspace_grid is not None:
            rospy.loginfo(f"C-Space calculated. Shape: {cspace_grid.shape}")
            # --- 发布 C-Space 到 RViz ---
            rospy.loginfo("Publishing C-Space grid to /cspace_map topic...")
            cspace_pub = rospy.Publisher('/cspace_map', OccupancyGrid, queue_size=1, latch=True)
            cspace_map_msg = OccupancyGrid()
            cspace_map_msg.header.stamp = rospy.Time.now()
            cspace_map_msg.header.frame_id = map_msg.header.frame_id # Use same frame as original map
            cspace_map_msg.info = map_info # Use same metadata as original map
            cspace_data = np.where(cspace_grid, 100, 0).astype(np.int8) # Convert boolean to 0/100
            cspace_map_msg.data = cspace_data.flatten().tolist()
            cspace_pub.publish(cspace_map_msg)
            rospy.loginfo(f"C-Space map published to /cspace_map (Frame: {cspace_map_msg.header.frame_id}).")
            # --- RViz 发布结束 ---

            # 添加调试打印 C-Space 值
            try:
                debug_indices = [(99, 99), (79, 19), (109, 109), (99, 179)] # (row, col)
                for r, c in debug_indices:
                    if 0 <= r < cspace_grid.shape[0] and 0 <= c < cspace_grid.shape[1]:
                         rospy.loginfo(f"DEBUG: Value of cspace_grid[{r}, {c}] is: {cspace_grid[r, c]}")
                    else:
                         rospy.logwarn(f"DEBUG: Index ({r}, {c}) is out of bounds for cspace_grid.")
            except IndexError:
                 rospy.logwarn("DEBUG: Index is out of bounds for cspace_grid during debug check.")

        else:
             rospy.logwarn("Failed to calculate C-Space. Using original occupancy grid for A*.")


        # --- 3. 准备输入数据 ---
        rospy.loginfo("Preparing SAMPLE ROI data and robot info...")

        # --- **** 使用你提供的新 sample_roi_list **** ---
        sample_roi_list = [
            {'id': 0, 'cells': '[(r,c) for r in range(61, 139) for c in range(61, 139)]', 'centroid_world': (0.000, 0.000)}, # 区域0 中心 (0.000, 0.000) 附近
            {'id': 1, 'cells': '[(r,c) for r in range(60, 139) for c in range(139, 197)]', 'centroid_world': (3.607, -0.030)}, # 区域1 中心 (3.607, -0.030) 附近
            {'id': 2, 'cells': '[(r,c) for r in range(3, 61) for c in range(60, 140)]', 'centroid_world': (-0.009, -3.620)}, # 区域2 中心 (-0.009, -3.620) 附近
            {'id': 3, 'cells': '[(r,c) for r in range(3, 60) for c in range(3, 60)]', 'centroid_world': (-3.566, -3.566)}, # 区域3 中心 (-3.566, -3.566) 附近
            {'id': 4, 'cells': '[(r,c) for r in range(139, 197) for c in range(4, 61)]', 'centroid_world': (-3.525, 3.554)}, # 区域4 中心 (-3.525, 3.554) 附近
            {'id': 5, 'cells': '[(r,c) for r in range(139, 197) for c in range(61, 139)]', 'centroid_world': (-0.011, 3.604)}, # 区域5 中心 (-0.011, 3.604) 附近
            {'id': 6, 'cells': '[(r,c) for r in range(139, 196) for c in range(139, 196)]', 'centroid_world': (3.528, 3.528)}, # 区域6 中心 (3.528, 3.528) 附近
            {'id': 7, 'cells': '[(r,c) for r in range(60, 139) for c in range(3, 61)]', 'centroid_world': (-3.607, -0.030)}, # 区域7 中心 (-3.607, -0.030) 附近
            {'id': 8, 'cells': '[(r,c) for r in range(3, 60) for c in range(139, 197)]', 'centroid_world': (3.557, -3.572)}, # 区域8 中心 (3.557, -3.572) 附近
        ]
        # --- **** sample_roi_list 替换结束 **** ---


        # --- 获取机器人信息 ---
        # **** 修正：确保机器人起始位姿在地图内且不在障碍物中 ****
        sample_robot_names = rospy.get_param("~robot_names", ['robot1', 'robot2', 'robot3']) # 保持3个机器人
        # **** MODIFIED: Use the hardcoded poses ****
        sample_robot_start_poses = {"robot1": (0.0, 0.0), "robot2": (-4.0, -1.0), "robot3": (4.0, 0.0)} # 默认起始位置 更改机器人位置为如上所示
        rospy.loginfo(f"Using hardcoded start poses: {sample_robot_start_poses}")

        # **** REMOVED: Parameter reading and walkability check loop ****
        # The following block is removed as poses are now hardcoded.
        # The walkability check will happen implicitly during A* cost calculation.
        # default_start_x = rospy.get_param("~default_start_x", -4.0)
        # default_start_y = rospy.get_param("~default_start_y", -4.0)
        # rospy.loginfo(f"Using robots: {sample_robot_names}")
        # temp_allocator_for_check = TaskAllocator(['dummy'], map_info, occupancy_grid, cspace_grid) # 临时实例用于检查
        # for i, name in enumerate(sample_robot_names):
        #      # 尝试从参数获取，否则使用偏移的默认值
        #      sx = rospy.get_param(f"~start_pose_{name}_x", default_start_x + i * 0.5) # 微小偏移
        #      sy = rospy.get_param(f"~start_pose_{name}_y", default_start_y + i * 0.5) # 微小偏移
        #      # 检查坐标是否有效
        #      start_rc_check = temp_allocator_for_check.world_to_grid(sx, sy)
        #      start_walkable_check = False
        #      if start_rc_check is not None:
        #           row_chk, col_chk = start_rc_check
        #           # **** 关键检查: 使用 cspace_grid (如果存在) ****
        #           grid_to_check = None
        #           grid_source = "None"
        #           if temp_allocator_for_check.cspace_grid is not None:
        #               grid_to_check = temp_allocator_for_check.cspace_grid
        #               grid_source = "C-Space"
        #               # 在 C-Space 中, False 是可走的 (非障碍)
        #               if 0 <= row_chk < grid_to_check.shape[0] and 0 <= col_chk < grid_to_check.shape[1]: # Bounds check
        #                   if not grid_to_check[row_chk, col_chk]:
        #                        start_walkable_check = True
        #               else:
        #                    rospy.logwarn(f"Start pose check for {name}: Grid index ({row_chk},{col_chk}) out of bounds for {grid_source}.")
        #                    start_walkable_check = False # Explicitly false if out of bounds
        #           elif temp_allocator_for_check.occupancy_grid is not None: # Fallback check
        #                grid_to_check = temp_allocator_for_check.occupancy_grid
        #                grid_source = "Occupancy"
        #                obstacle_threshold_chk = rospy.get_param("~allocation_obstacle_threshold", 50)
        #                # 在 Occupancy Grid 中, < threshold 且 != -1 是可走的
        #                if 0 <= row_chk < grid_to_check.shape[0] and 0 <= col_chk < grid_to_check.shape[1]: # Bounds check
        #                    if not (grid_to_check[row_chk, col_chk] >= obstacle_threshold_chk or grid_to_check[row_chk, col_chk] == -1):
        #                         start_walkable_check = True
        #                else:
        #                     rospy.logwarn(f"Start pose check for {name}: Grid index ({row_chk},{col_chk}) out of bounds for {grid_source}.")
        #                     start_walkable_check = False # Explicitly false if out of bounds
        #           else: # No grid to check against, assume walkable but warn
        #                start_walkable_check = True
        #                rospy.logwarn(f"Cannot check walkability for robot {name} start pose {sx, sy} - no grid data.")
        #           # 打印检查结果
        #           rospy.loginfo(f"  Checking start pose for {name}: World ({sx:.2f}, {sy:.2f}) -> Grid ({row_chk}, {col_chk}). Source: {grid_source}. Walkable: {start_walkable_check}")
        #           if grid_to_check is not None and (0 <= row_chk < grid_to_check.shape[0] and 0 <= col_chk < grid_to_check.shape[1]):
        #                rospy.loginfo(f"   -> Value at grid[{row_chk}, {col_chk}]: {grid_to_check[row_chk, col_chk]}")
        #           elif grid_to_check is not None:
        #                rospy.loginfo(f"   -> Index ({row_chk},{col_chk}) out of bounds for grid check.")
        #      if start_rc_check is None or not start_walkable_check:
        #           rospy.logfatal(f"FATAL: Start pose ({sx:.2f}, {sy:.2f}) for robot '{name}' is outside map or NOT walkable in {grid_source}! Please provide valid start poses via parameters.")
        #           sys.exit(1) # 强制退出，因为无效起始点会导致必然失败
        #      else:
        #           sample_robot_start_poses[name] = (sx, sy)
        # del temp_allocator_for_check # 删除临时实例


        # --- 4. 创建 TaskAllocator 实例 ---
        rospy.loginfo("Creating TaskAllocator instance...")
        allocator = TaskAllocator(
            robot_names=sample_robot_names,
            map_info=map_info,
            occupancy_grid=occupancy_grid,
            cspace_grid=cspace_grid # 传入 C-Space
        )

        # --- 5. 执行任务分配 ---
        assignment_result = None
        if not sample_robot_names:
             rospy.logwarn("No robots defined. Skipping allocation.")
             assignment_result = {}
        else:
             rospy.loginfo("Starting task allocation process...")
             # **** Pass the hardcoded poses ****
             assignment_result = allocator.allocate_tasks(
                 sample_roi_list,
                 sample_robot_start_poses # Use the hardcoded dictionary
             )
             if assignment_result is None:
                  rospy.logerr("Task allocation failed and returned None.")
                  assignment_result = {}


        # --- 6. 打印最终分配结果 ---
        rospy.loginfo("\n========= Final Task Assignment (GA-mTSP-A*) =========")
        if not assignment_result and sample_robot_names:
             rospy.logwarn("Allocation resulted in empty assignment or failed.")
             for robot in sample_robot_names: print(f"Robot {robot} assigned ROIs (in order): []")
        elif not sample_robot_names:
             rospy.logwarn("No robots defined.")
        else:
             all_assigned_roi_ids = set()
             for robot, assigned_rois in assignment_result.items():
                  ordered_ids = [roi['id'] for roi in assigned_rois]
                  all_assigned_roi_ids.update(ordered_ids)
                  print(f"Robot {robot} assigned ROIs (in order): {ordered_ids}")

             # 检查是否有有效 ROI 未被分配
             if hasattr(allocator, 'roi_map'):
                 valid_roi_ids_after_precompute = set(allocator.roi_map.keys())
                 unassigned_valid_rois = valid_roi_ids_after_precompute - all_assigned_roi_ids
                 if unassigned_valid_rois:
                      rospy.logwarn(f"Warning: Valid ROIs not assigned: {sorted(list(unassigned_valid_rois))}")

        rospy.loginfo("========================================================")


    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down.")
    except ImportError as e:
         rospy.logfatal(f"Import Error: {e}. Ensure required libraries are installed.")
    except Exception as e:
        rospy.logerr("Unexpected error in main execution block:")
        rospy.logerr(traceback.format_exc())
    finally:
         rospy.loginfo("Task allocator node finished.")


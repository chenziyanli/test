#!/usr/bin/env python3
import rospy
import numpy as np
import math
import random
import traceback
import sys
from functools import partial
from collections import defaultdict
import time # 用于计时

# ROS Msgs/Srvs
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose

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

# --- 辅助函数：C-Space 计算 (如果需要在此脚本内计算) ---
def calculate_cspace(occupancy_grid, map_resolution, robot_radius_m, obstacle_threshold=50):
    """(辅助函数) 计算配置空间"""
    if not _SCIPY_AVAILABLE or occupancy_grid is None or map_resolution <= 0:
        rospy.logwarn("Cannot calculate C-Space: SciPy not available, grid missing, or invalid resolution.")
        return None

    if robot_radius_m <= 0:
        rospy.logwarn("Robot radius is zero or negative, using original occupancy grid as C-Space.")
        # 将障碍物和未知区域标记为 True (障碍)
        return (occupancy_grid >= obstacle_threshold) | (occupancy_grid == -1)

    robot_radius_cells = int(math.ceil(robot_radius_m / map_resolution))
    rospy.loginfo(f"Calculating C-Space with radius {robot_radius_m}m ({robot_radius_cells} cells)...")

    obstacle_mask = (occupancy_grid >= obstacle_threshold) | (occupancy_grid == -1)
    struct = generate_binary_structure(2, 2) # 8-connectivity

    try:
        start_time = time.time()
        cspace_grid = binary_dilation(obstacle_mask, structure=struct, iterations=robot_radius_cells)
        end_time = time.time()
        rospy.loginfo(f"cspace_grid {cspace_grid}")
        rospy.loginfo(f"C-Space calculation took {end_time - start_time:.3f} seconds.")
        return cspace_grid # 返回布尔型 NumPy 数组 (True=Obstacle)
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

        # 地图信息 (从外部传入或通过方法设置)
        self.map_info = map_info
        self.occupancy_grid = occupancy_grid
        self.cspace_grid = cspace_grid # 优先使用 C-Space
        self.map_resolution = map_info.resolution if map_info else None
        self.map_origin = map_info.origin if map_info else None
        self.map_width = map_info.width if map_info else 0
        self.map_height = map_info.height if map_info else 0
        self._map_ready = self._validate_map_data()

        # DEAP 创建器 (确保只创建一次)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # 最小化 Makespan
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # 预计算成本的存储
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
           return False
        # 检查维度是否匹配
        if self.occupancy_grid.shape != (self.map_height, self.map_width):
             rospy.logerr(f"Occupancy grid shape {self.occupancy_grid.shape} does not match map info ({self.map_height}, {self.map_width})")
             return False
        if self.cspace_grid is not None and self.cspace_grid.shape != (self.map_height, self.map_width):
             rospy.logerr(f"C-Space grid shape {self.cspace_grid.shape} does not match map info ({self.map_height}, {self.map_width})")
             return False # C-Space 必须与地图匹配
        return True

    def is_ready(self):
        """检查分配器是否有有效地图"""
        return self._map_ready

    def grid_to_world(self, row, col):
        """栅格坐标到世界坐标"""
        if not self.is_ready(): return None, None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        wx = ox + (col + 0.5) * self.map_resolution
        wy = oy + (row + 0.5) * self.map_resolution
        return wx, wy

    def world_to_grid(self, wx, wy):
        """世界坐标到栅格坐标"""
        if not self.is_ready() or self.map_resolution == 0: return None, None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        col = int((wx - ox) / self.map_resolution)
        row = int((wy - oy) / self.map_resolution)
        if 0 <= row < self.map_height and 0 <= col < self.map_width:
            return row, col
        return None, None

    def _estimate_coverage_cost(self, roi, coverage_speed=0.1, coverage_width=0.2):
        """估算覆盖 ROI 的时间成本"""
        # ... (与上个文件中的 _estimate_coverage_cost 相同, 使用 self.map_resolution) ...
        coverage_speed = rospy.get_param("~allocation_coverage_speed", coverage_speed)
        coverage_width = rospy.get_param("~allocation_coverage_width", coverage_width)
        if not roi or 'cells' not in roi or not roi['cells'] or coverage_speed <= 0 or coverage_width <= 0:
            return 0.0
        if self.map_resolution is None or self.map_resolution <= 0: return 0.0
        area = len(roi['cells']) * (self.map_resolution ** 2)
        coverage_time = area / (coverage_speed * coverage_width)
        return coverage_time

    def _calculate_astar_path_cost(self, start_rc, end_rc):
        """使用 A* 算法计算两栅格点间的路径长度成本"""
        if not self.is_ready(): return float('inf')
        if start_rc is None or end_rc is None: return float('inf')
        if start_rc == end_rc: return 0.0

        # --- 选择地图数据 (优先 C-Space) ---
        grid_matrix = None
        if self.cspace_grid is not None:
            # C-Space: True 是障碍 -> 1, False 是空闲 -> 0
            grid_matrix = np.where(self.cspace_grid, 0, 1).astype(np.int32)
            # rospy.logdebug("Using C-Space grid for A*")
        elif self.occupancy_grid is not None:
            rospy.logwarn_throttle(10, "C-Space grid not available, falling back to occupancy grid for A*.")
            obstacle_threshold = rospy.get_param("~allocation_obstacle_threshold", 50)
            grid_matrix = np.where(
                (self.occupancy_grid >= obstacle_threshold) | (self.occupancy_grid == -1), 1, 0
            ).astype(np.int32)
        else:
            rospy.logerr("A* Cost Calc: No grid data available.")
            return float('inf')

        # --- 创建 pathfinding Grid ---
        try:
            # Transpose matrix because pathfinding library expects matrix[x][y] (col, row)
            grid = Grid(matrix=grid_matrix.T)
            start_node = grid.node(start_rc[1], start_rc[0]) # node(col, row)
            end_node = grid.node(end_rc[1], end_rc[0])   # node(col, row)
        except Exception as e:
            rospy.logerr(f"Failed to create pathfinding grid or nodes from ({start_rc}) to ({end_rc}): {e}")
            return float('inf') # Return high cost if nodes are invalid (e.g., on obstacle)

        # --- 检查起点终点是否可通行 ---
        # (重要: 如果起点或终点本身在障碍物上，规划会失败)
        if not grid.walkable(start_node.x, start_node.y):
             rospy.logwarn(f"A* Start node {start_rc} (col={start_node.x},row={start_node.y}) is not walkable!")
             return float('inf')
        if not grid.walkable(end_node.x, end_node.y):
             rospy.logwarn(f"A* End node {end_rc} (col={end_node.x},row={end_node.y}) is not walkable!")
             return float('inf')


        # --- 运行 A* ---
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        try:
            path_rc_tuples, runs = finder.find_path(start_node, end_node, grid)
            if not path_rc_tuples:
                rospy.logwarn(f"A* found no path from {start_rc} to {end_rc}")
                return float('inf')

            # --- 计算路径长度 ---
            path_length_m = 0.0
            for i in range(len(path_rc_tuples) - 1):
                # path_rc_tuples contains (col, row)
                c1, r1 = path_rc_tuples[i]
                c2, r2 = path_rc_tuples[i+1]
                # Use Euclidean distance between cell centers
                w1_x, w1_y = self.grid_to_world(r1, c1)
                w2_x, w2_y = self.grid_to_world(r2, c2)
                if w1_x is not None and w2_x is not None:
                    try: dist = math.dist((w1_x, w1_y), (w2_x, w2_y))
                    except AttributeError: dist = math.sqrt((w1_x-w2_x)**2 + (w1_y-w2_y)**2)
                    path_length_m += dist
                else:
                    return float('inf') # Failed conversion

            # 返回距离，确保非零
            return path_length_m if path_length_m > 1e-3 else 1e-3

        except Exception as e:
            rospy.logerr(f"Error during A* path finding from {start_rc} to {end_rc}: {e}")
            return float('inf')

    def _precompute_costs(self, roi_list, robot_start_poses_map):
        """预计算所有旅行成本(使用A*)和覆盖成本"""
        rospy.loginfo("Precomputing costs using A*...")
        if not self.is_ready() or not roi_list:
             rospy.logerr("Map not ready or ROI list empty for precomputation.")
             return False

        # 1. Get valid ROIs and representative points (centroids)
        valid_rois = []
        roi_points = {}
        self.roi_map = {}
        for roi in roi_list:
            roi_id = roi.get('id')
            centroid = roi.get('centroid_world')
            cells = roi.get('cells')
            if roi_id is not None and centroid is not None and cells:
                valid_rois.append(roi)
                roi_points[roi_id] = centroid
                self.roi_map[roi_id] = roi
            else:
                 rospy.logwarn(f"ROI missing id, centroid_world, or cells, skipping: {roi_id if roi_id is not None else 'No ID'}")
        # rospy.loginfo(f"roi_points: {roi_points}")
        if not valid_rois:
            rospy.logerr("No valid ROIs for cost precomputation.")
            return False

        self.num_valid_rois = len(valid_rois)
        valid_roi_ids = sorted(self.roi_map.keys())
        self.roi_idx_to_id = {i: roi_id for i, roi_id in enumerate(valid_roi_ids)}

        # 2. Estimate coverage costs
        self.coverage_costs = {}
        for roi_id in valid_roi_ids:
            self.coverage_costs[roi_id] = self._estimate_coverage_cost(self.roi_map[roi_id], self.map_resolution)
            rospy.loginfo(f"  Coverage cost ROI {roi_id}: {self.coverage_costs[roi_id]:.2f}")

        # 3. Precompute travel costs using A*
        self.travel_costs = defaultdict(dict)
        self.start_costs = defaultdict(dict)
        rospy.loginfo(f"Calculating A* travel costs ({self.num_robots} robots, {self.num_valid_rois} ROIs)...")
        start_time_astar = time.time()

        #   a) From robot starts to ROIs
        for r_name in self.robot_names:
            start_pose_xy = robot_start_poses_map.get(r_name, (0.0, 0.0))
            start_rc = self.world_to_grid(start_pose_xy[0], start_pose_xy[1])
            if start_rc is None:
                rospy.logwarn(f"Start pose {start_pose_xy} for robot {r_name} outside map. Assigning inf cost.")
                for roi_id in valid_roi_ids: self.start_costs[r_name][roi_id] = float('inf')
                continue
            for roi_id in valid_roi_ids:
                 roi_start_point_xy = roi_points[roi_id]
                 end_rc = self.world_to_grid(roi_start_point_xy[0], roi_start_point_xy[1])
                 rospy.loginfo(f"  A* start cost from {r_name} to ROI {roi_id}: {start_rc} -> {end_rc}")
                 if end_rc is None: cost = float('inf')
                 else: cost = self._calculate_astar_path_cost(start_rc, end_rc)
                 self.start_costs[r_name][roi_id] = cost
            rospy.loginfo(f" Calculated A* start costs for {r_name}.")


        #   b) Between ROIs
        num_pairs = len(valid_roi_ids) * (len(valid_roi_ids) - 1)
        count = 0
        rospy.loginfo(f" Calculating {num_pairs} inter-ROI A* travel costs...")
        for i, roi_id_i in enumerate(valid_roi_ids):
            point_i_xy = roi_points[roi_id_i]
            start_rc_i = self.world_to_grid(point_i_xy[0], point_i_xy[1])
            if start_rc_i is None:
                for roi_id_j in valid_roi_ids: self.travel_costs[roi_id_i][roi_id_j] = float('inf')
                continue
            for j, roi_id_j in enumerate(valid_roi_ids):
                if i == j:
                    self.travel_costs[roi_id_i][roi_id_j] = 0.0
                    continue
                point_j_xy = roi_points[roi_id_j]
                end_rc_j = self.world_to_grid(point_j_xy[0], point_j_xy[1])
                if end_rc_j is None: cost = float('inf')
                else: cost = self._calculate_astar_path_cost(start_rc_i, end_rc_j)
                self.travel_costs[roi_id_i][roi_id_j] = cost
                count += 1
                if count % 20 == 0 or count == num_pairs:
                     rospy.loginfo(f"  Calculated A* travel costs: {count}/{num_pairs}")
            # rospy.loginfo(f" Calculated A* travel costs from ROI {roi_id_i} ({i+1}/{len(valid_roi_ids)}).")


        end_time_astar = time.time()
        rospy.loginfo(f"Finished precomputing A* costs in {end_time_astar - start_time_astar:.3f} seconds.")
        return True

    def _decode_chromosome(self, individual):
        """解码 DEAP 个体为每个机器人的 ROI 索引序列"""
        # Uses self.num_valid_rois and self.num_robots
        tours = []
        current_tour = []
        separator_indices = set(range(self.num_valid_rois, self.num_valid_rois + self.num_robots - 1))
        for item in individual:
            if item in separator_indices:
                if current_tour: tours.append(current_tour)
                current_tour = []
            else:
                current_tour.append(item) # item is an ROI index
        if current_tour: tours.append(current_tour)
        while len(tours) < self.num_robots: tours.append([])
        return tours[:self.num_robots]

    def _evaluate_makespan(self, individual, robot_start_poses_map):
        """DEAP 适应度评估函数：计算 Makespan (使用预计算的A*成本)"""
        if not self.roi_map: return (float('inf'),)

        robot_roi_indices_tours = self._decode_chromosome(individual)
        robot_completion_times = []

        for i, tour_indices in enumerate(robot_roi_indices_tours):
            robot_name = self.robot_names[i]
            current_robot_time = 0.0
            last_roi_id = None # Start from depot

            if not tour_indices:
                 robot_completion_times.append(current_robot_time)
                 continue

            # Travel from start to first ROI
            first_roi_idx = tour_indices[0]
            first_roi_id = self.roi_idx_to_id.get(first_roi_idx)
            if first_roi_id is None: return (float('inf'),)
            current_robot_time += self.start_costs[robot_name].get(first_roi_id, float('inf'))

            # Add coverage and travel times for the tour
            for j, current_roi_idx in enumerate(tour_indices):
                current_roi_id = self.roi_idx_to_id.get(current_roi_idx)
                if current_roi_id is None: return (float('inf'),)

                # Add travel time from previous ROI (or start)
                if j > 0:
                    last_roi_id = self.roi_idx_to_id.get(tour_indices[j-1])
                    if last_roi_id is None: return (float('inf'),)
                    current_robot_time += self.travel_costs[last_roi_id].get(current_roi_id, float('inf'))

                # Add coverage time
                current_robot_time += self.coverage_costs.get(current_roi_id, 0.0)

            robot_completion_times.append(current_robot_time)

        makespan = max(robot_completion_times) if robot_completion_times else 0.0
        return (makespan,) # Return tuple

    def allocate_tasks(self, roi_list, robot_start_poses_map):
        """主分配函数，运行 GA 求解"""
        if not self.is_ready():
             rospy.logerr("Allocator is not ready (no map data). Cannot allocate.")
             return {}

        rospy.loginfo("Starting Task Allocation using GA for mTSP (A* Costs)...")

        # --- 1. 预计算成本 ---
        if not self._precompute_costs(roi_list, robot_start_poses_map):
            rospy.logerr("Task allocation failed due to cost precomputation error.")
            return {}
        if self.num_valid_rois == 0:
             rospy.logwarn("No valid ROIs to allocate after precomputation.")
             return {}

        # --- 2. 设置 DEAP 工具箱 ---
        toolbox = base.Toolbox()

        # Chromosome definition
        num_separators = self.num_robots - 1
        elements = list(range(self.num_valid_rois)) + list(range(self.num_valid_rois, self.num_valid_rois + num_separators))
        toolbox.register("indices", random.sample, elements, len(elements))
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register operators
        toolbox.register("evaluate", self._evaluate_makespan, robot_start_poses_map=robot_start_poses_map)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # --- 3. 运行遗传算法 ---
        population_size = rospy.get_param("~ga_population_size", 100)
        num_generations = rospy.get_param("~ga_num_generations", 80)
        crossover_prob = rospy.get_param("~ga_crossover_prob", 0.8)
        mutation_prob = rospy.get_param("~ga_mutation_prob", 0.3)

        rospy.loginfo(f"Running GA: Pop={population_size}, Gen={num_generations}, CXPB={crossover_prob}, MUTPB={mutation_prob}")
        start_ga_time = time.time()

        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean); stats.register("min", np.min); stats.register("max", np.max)

        try:
             algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                                 ngen=num_generations, stats=stats, halloffame=hof, verbose=True)
        except Exception as e:
             rospy.logerr(f"Error during GA execution: {e}"); return {}

        end_ga_time = time.time()
        rospy.loginfo(f"GA execution took {end_ga_time - start_ga_time:.3f} seconds.")

        # --- 4. 处理结果 ---
        if not hof: rospy.logerr("GA Hall of Fame is empty."); return {}

        best_individual = hof[0]
        best_makespan = best_individual.fitness.values[0]
        rospy.loginfo(f"GA finished. Best solution Makespan estimate: {best_makespan:.2f}")

        best_tours_indices = self._decode_chromosome(best_individual)

        final_assignment = defaultdict(list)
        for i, tour_indices in enumerate(best_tours_indices):
            robot_name = self.robot_names[i]
            for roi_idx in tour_indices:
                roi_id = self.roi_idx_to_id.get(roi_idx)
                if roi_id is not None and roi_id in self.roi_map:
                    final_assignment[robot_name].append(self.roi_map[roi_id])
                else:
                     rospy.logwarn(f"Invalid ROI index/id {roi_idx}/{roi_id} in best tour for {robot_name}")

        rospy.loginfo("--- Final Task Allocation (Robot -> Ordered ROI IDs) ---")
        total_assigned_rois = 0
        for robot, rois in final_assignment.items():
            roi_ids_ordered = [r['id'] for r in rois]
            rospy.loginfo(f"  {robot}: Assigned ROIs {roi_ids_ordered}")
            total_assigned_rois += len(rois)
        rospy.loginfo(f"Total ROIs assigned: {total_assigned_rois} / {self.num_valid_rois}")
        rospy.loginfo("-----------------------------------------------------")


        return dict(final_assignment)


# ==============================================================================
# Main Execution Block (示例用法)
# ==============================================================================
if __name__ == '__main__':
    # --- 检查依赖 ---
    if not _DEAP_AVAILABLE or not _PATHFINDING_AVAILABLE:
        rospy.logfatal("Missing required libraries (DEAP and/or python-pathfinding). Please install.")
        sys.exit(1)

    try:
        rospy.init_node('task_allocator_ga_mtsp_astar_node', anonymous=True)

        # --- 1. 获取地图数据 (需要先运行 map_server) ---
        map_msg = None
        map_topic = rospy.get_param("~map_topic", "/map")
        rospy.loginfo(f"Waiting for map on {map_topic}...")
        try:
            map_msg = rospy.wait_for_message(map_topic, OccupancyGrid, timeout=15.0)
            rospy.loginfo("Map received.")
        except rospy.ROSException as e:
            rospy.logfatal(f"Failed to receive map: {e}. Exiting.")
            sys.exit(1)
        except Exception as e:
             rospy.logfatal(f"Error receiving map: {e}"); sys.exit(1)

        map_info = map_msg.info
        occupancy_grid = np.array(map_msg.data).reshape((map_info.height, map_info.width)).astype(np.int8)

        # --- 2. 计算 C-Space (重要!) ---
        robot_radius = rospy.get_param("~robot_radius", 0.05) # Use a realistic radius
        obstacle_thresh = rospy.get_param("~obstacle_threshold", 50)
        cspace_grid = calculate_cspace(occupancy_grid, map_info.resolution, robot_radius, obstacle_thresh)
        rospy.loginfo(f"Occupancy grid size: {occupancy_grid.shape}, C-Space grid size: {cspace_grid.shape if cspace_grid is not None else 'N/A'}")
        if cspace_grid is None:
             rospy.logwarn("Failed to calculate C-Space. A* will use occupancy grid (less accurate).")
             # 可以选择退出，或者让 A* 使用 occupancy_grid 回退
        # --- 3. 准备输入数据 (示例 ROI 和机器人) ---
        # !! 重要: 这里的 ROI 列表应该由您的 AreaDecomposer 节点生成并传递过来 !!
        # !! 'centroid_world' 和 'cells' 字段是必需的 !!
        rospy.loginfo("Using SAMPLE ROI data for testing.")
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
        # 过滤掉质心在地图外的 ROI (如果需要)
        temp_allocator_for_check = TaskAllocator(['dummy'], map_info, occupancy_grid, cspace_grid) #临时实例
        valid_sample_roi_list = []
        for roi in sample_roi_list:
            grid_coords = temp_allocator_for_check.world_to_grid(roi['centroid_world'][0], roi['centroid_world'][1])
            if grid_coords is not None:
                valid_sample_roi_list.append(roi)
            else:
                rospy.logwarn(f"Sample ROI {roi['id']} centroid is outside map, removing.")
        del temp_allocator_for_check # 删除临时实例


        # sample_robot_names = rospy.get_param("~robot_names", ['robot1', 'robot2','robot3']) # 从参数获取
        # sample_robot_start_poses = {"robot1": (0.0, 0.0), "robot2": (-4.0, -1.0), "robot3": (4.0, 0.0)} # 默认起始位置
        sample_robot_names = rospy.get_param("~robot_names", ['robot1']) # 从参数获取
        sample_robot_start_poses = {"robot1": (0.0, 0.0), } # 默认起始位置
        # --- 4. 创建分配器实例 (传入地图数据) ---
        allocator = TaskAllocator(
            robot_names=sample_robot_names,
            map_info=map_info,
            occupancy_grid=occupancy_grid,
            cspace_grid=cspace_grid # 传入 C-Space!
        )

        # --- 5. 执行分配 ---
        rospy.loginfo("Starting task allocation process...")
        assignment_result = allocator.allocate_tasks(
            valid_sample_roi_list, # 使用过滤后的 ROI
            sample_robot_start_poses
            # map_resolution is now stored in allocator instance
        )

        # --- 6. 打印结果 ---
        if assignment_result:
            rospy.loginfo("\n========= Final Assignment (Using A*) =========")
            for robot, assigned_rois in assignment_result.items():
                ordered_ids = [roi['id'] for roi in assigned_rois]
                print(f"Robot {robot} assigned ROIs (in order): {ordered_ids}")
            rospy.loginfo("================================================")
        else:
            rospy.logerr("Task allocation process failed.")

    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received.")
    except ImportError as e:
         rospy.logfatal(f"Import Error: {e}. Make sure required libraries (DEAP, pathfinding, scipy) are installed.")
    except Exception as e:
        rospy.logerr("An unexpected error occurred in the Task Allocator tester.")
        rospy.logerr(traceback.format_exc())
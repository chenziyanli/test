#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose,PoseArray, PoseStamped,Quaternion# Added PoseStamped for path publishing
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Path                 # Added Path for publishing
import numpy as np
import traceback
import math
try:
    # Needed for C-Space calculation (obstacle inflation)
    from scipy.ndimage import binary_dilation, generate_binary_structure
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    rospy.logwarn("CoveragePlanner: scipy.ndimage not found. C-Space calculation will be unavailable.")

class CoveragePlanner:
    def __init__(self):
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = 0
        self.map_height = 0
        self.occupancy_grid = None
        self.cspace_grid = None # Configuration space grid
        self._map_received = False
        self._cspace_calculated = False
        self._robot_radius_cells_used_for_cspace = -1 # Track C-space params
        self.map_info = None
        self.map_frame_id = "map" # 提供一个默认值或设为 None
        self.cspace_pub = rospy.Publisher('/cspace_map', OccupancyGrid, queue_size=1, latch=True)

        # Subscribe to map topic
        map_topic = rospy.get_param("~map_topic", "/map") # Allow map topic override via param
        try:
            rospy.loginfo("CoveragePlanner: Waiting for map topic [%s]...", map_topic)
            map_msg = rospy.wait_for_message(map_topic, OccupancyGrid, timeout=rospy.Duration(20.0))
            self._process_map_data(map_msg)
            rospy.loginfo("CoveragePlanner: Map received and processed.")
        except rospy.ROSException as e:
            rospy.logerr(f"CoveragePlanner: Failed to receive map from {map_topic}: {e}")
        except Exception as e:
            rospy.logerr(f"CoveragePlanner: Error during initialization: {e}")
            rospy.logerr(traceback.format_exc())

    def _process_map_data(self, msg):
        """Processes the received OccupancyGrid message."""
        if msg.info.resolution == 0:
            rospy.logerr("Received map with zero resolution. Cannot use this map.")
            return
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_info = msg.info # 保存地图元数据以供发布C-Space时使用
        self.map_frame_id = msg.header.frame_id # <--- 添加这行，保存 frame_id
        # Reshape data correctly: row-major order means (height, width)
        self.map_data = np.array(msg.data).reshape((self.map_height, self.map_width))
        
        # Ensure int type for indexing if needed, though values are occupancy
        self.occupancy_grid = self.map_data.astype(np.int8)
        self._map_received = True
        self._cspace_calculated = False # Reset C-space status when map updates
        rospy.loginfo(f"CoveragePlanner: Map processed. Shape: {self.occupancy_grid.shape}, Resolution: {self.map_resolution}")

    def _calculate_cspace(self, robot_radius_m, obstacle_threshold=1):
        """Calculates Configuration Space by inflating obstacles based on robot radius."""
        if not self.is_ready():
            rospy.logerr("Cannot calculate C-Space: Map not ready.")
            return False
        if not _SCIPY_AVAILABLE:
            rospy.logerr("Cannot calculate C-Space: SciPy not available.")
            return False

        # Convert radius to cells, ensuring it's at least 0
        # Using math.ceil ensures we inflate by enough cells even for small radii
        robot_radius_cells = 0
        if robot_radius_m > 0 and self.map_resolution > 0:
             robot_radius_cells = int(math.ceil(robot_radius_m / self.map_resolution))
        elif robot_radius_m > 0:
             rospy.logwarn("Map resolution is zero, cannot calculate robot radius in cells.")
             return False # Cannot calculate C-Space if resolution is zero


        rospy.loginfo(f"Calculating C-Space with robot radius {robot_radius_m}m ({robot_radius_cells} cells)...")

        # Check if C-space with the same radius is already computed
        if self._cspace_calculated and self._robot_radius_cells_used_for_cspace == robot_radius_cells:
             rospy.loginfo("Using previously computed C-Space.")
             return True

        # If radius is 0, C-space is just the original obstacles
        if robot_radius_cells == 0:
            rospy.loginfo("Robot radius is 0 cells, using original obstacles/unknown as C-Space.")
            self.cspace_grid = (self.occupancy_grid >= obstacle_threshold) | (self.occupancy_grid == -1)
            self._robot_radius_cells_used_for_cspace = 0
            self._cspace_calculated = True
            return True

        # Define obstacles: values >= threshold or unknown (-1)
        obstacle_mask = (self.occupancy_grid >= obstacle_threshold) | (self.occupancy_grid == -1)

        # Define the structuring element for dilation (connectivity)
        # struct = generate_binary_structure(2, 1)  # 4-connectivity
        struct = generate_binary_structure(2, 2)  # 8-connectivity (more conservative)

        # Dilate the obstacles
        try:
            self.cspace_grid = binary_dilation(obstacle_mask, structure=struct, iterations=robot_radius_cells)
            self._robot_radius_cells_used_for_cspace = robot_radius_cells
            self._cspace_calculated = True
            rospy.loginfo("C-Space calculated successfully.")
            return True
        except Exception as e:
            rospy.logerr(f"Error during C-Space calculation (binary_dilation): {e}")
            rospy.logerr(traceback.format_exc())
            self._cspace_calculated = False
            return False

    def is_ready(self):
        """Checks if the planner is ready (has received map data)."""
        return self._map_received

    # --- Coordinate Conversion Functions ---
    def world_to_grid(self, wx, wy):
        """Converts world coordinates (meters) to grid cell coordinates (row, col)."""
        if not self.is_ready() or self.map_resolution == 0: return None, None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        # Assume no rotation for simplicity.
        col = int((wx - ox) / self.map_resolution)
        row = int((wy - oy) / self.map_resolution)
        # Check bounds before returning
        if 0 <= row < self.map_height and 0 <= col < self.map_width:
            return row, col
        # Return None for row and col if out of bounds
        # Useful to check as `if r is None or c is None:`
        return None, None

    def grid_to_world(self, row, col):
        """Converts grid cell coordinates (row, col) to world coordinates (meters, center of cell)."""
        if not self.is_ready(): return None, None
        ox = self.map_origin.position.x
        oy = self.map_origin.position.y
        # Calculate center of the grid cell
        wx = ox + (col + 0.5) * self.map_resolution
        wy = oy + (row + 0.5) * self.map_resolution
        return wx, wy

    # --- Cell Check Function ---
    def is_cell_free_in_cspace_roi(self, row, col, roi):
        """
        Checks if the grid cell (row, col) is within the ROI (world coordinates)
        and is considered free in the calculated Configuration Space.

        Args:
            row (int): Grid row index.
            col (int): Grid column index.
            roi (dict): Region of Interest {'min_x', 'min_y', 'max_x', 'max_y'}.

        Returns:
            bool: True if the cell is within bounds, within ROI, and free in C-Space.
        """
        # 1. Check C-Space availability and grid bounds
        if not self._cspace_calculated:
             rospy.logerr_throttle(10,"C-Space not calculated, cannot check cell freeness.")
             return False
        if not (0 <= row < self.map_height and 0 <= col < self.map_width):
            # This check might be redundant if called carefully, but safe to keep
            return False

        # 2. Check C-Space grid (True means inflated obstacle)
        if self.cspace_grid[row, col]:
            return False # Cell is blocked in C-Space
        # cspace_blocked = self.cspace_grid[row, col]
        # rospy.logwarn(f"Checking cell ({row}, {col}). C-Space Blocked: {cspace_blocked}") # 添加日志

        # if cspace_blocked:
        #     rospy.logwarn(f"  --> Cell ({row}, {col}) RETURN False (C-Space Blocked)") # 添加日志
        # return False
        # 3. Check if cell center is within world ROI
        wx, wy = self.grid_to_world(row, col)
        # Need robust check in case grid_to_world failed somehow (shouldn't if bounds ok)
        if wx is None or wy is None:
             rospy.logwarn_throttle(10, f"Failed to convert supposedly valid grid cell ({row},{col}) to world.")
             return False
        if not (roi['min_x'] <= wx < roi['max_x'] and roi['min_y'] <= wy < roi['max_y']):
             return False # Cell center not within the defined ROI rectangle

        # If all checks pass, the cell is considered free for the robot's center
        return True


    def publish_cspace_map(self):
        """将计算出的 C-Space 地图发布为 OccupancyGrid 消息"""
        if not self.is_ready() or self.cspace_grid is None :
            rospy.logwarn("Cannot publish C-Space map: Planner not ready, C-Space not calculated, or map_info missing.")
            return

        # 创建一个新的 OccupancyGrid 消息
        cspace_msg = OccupancyGrid()
        cspace_msg.header.stamp = rospy.Time.now()
        # 使用原始地图的 frame_id
        cspace_msg.header.frame_id = self.map_frame_id # 使用保存的 frame_id
        # 使用原始地图的 info (分辨率, 宽高, 原点等)
        cspace_msg.info = self.map_info

        # 将布尔型的 cspace_grid 转换为 OccupancyGrid 的 data 格式 (0 代表空闲, 100 代表障碍物)
        # np.where(condition, value_if_true, value_if_false)
        cspace_data = np.where(self.cspace_grid, 100, 0).astype(np.int8)

        # 确保数据长度正确
        expected_length = self.map_info.width * self.map_info.height
        if len(cspace_data.flatten()) != expected_length:
             rospy.logerr(f"C-Space data length mismatch! Expected {expected_length}, Got {len(cspace_data.flatten())}")
             return

        # 将二维数组展平为一维列表
        cspace_msg.data = cspace_data.flatten().tolist()

        # 发布消息
        try:
            self.cspace_pub.publish(cspace_msg)
            rospy.loginfo("Published C-Space map to /cspace_map")
        except Exception as e:
            rospy.logerr(f"Failed to publish C-Space map: {e}")
    # --- 发布函数结束 ---
    # --- Main Planning Algorithm ---
    # --- Main Planning Algorithm (with pre-append checks) ---
    def plan_boustrophedon_path(self, roi, robot_radius, coverage_width, overlap_ratio=0.1, obstacle_threshold=50):
        """
        Plans a Boustrophedon path within the ROI, considering robot radius (via C-Space)
        and coverage width (for row stepping). Includes pre-append checks for debugging.

        Args:
            roi (dict): {'min_x': float, 'min_y': float, 'max_x': float, 'max_y': float}
            robot_radius (float): Robot radius for C-Space calculation (meters).
            coverage_width (float): Robot's effective coverage width (meters).
            overlap_ratio (float): Desired overlap between adjacent rows (0.0 to < 1.0).
            obstacle_threshold (int): Occupancy grid value above which cells are considered obstacles.

        Returns:
            list: List of waypoints [(x1, y1), (x2, y2), ...] in world coordinates. Empty list on failure.
        """
        # 1. Initial Checks
        if not self.is_ready():
            rospy.logerr("Planner not ready (no map). Cannot plan path.")
            return []
        if not _SCIPY_AVAILABLE:
             rospy.logerr("SciPy not available, cannot perform C-Space calculation needed for planning.")
             return []
        if coverage_width <= 0 or robot_radius < 0 or not (0.0 <= overlap_ratio < 1.0):
             rospy.logerr(f"Invalid parameters: coverage_width={coverage_width} (>0 needed), "
                          f"robot_radius={robot_radius} (>=0 needed), overlap_ratio={overlap_ratio} ([0,1) needed).")
             return []
        if self.map_resolution <= 0:
             rospy.logerr("Map resolution is zero or negative. Cannot plan.")
             return []

        # 2. Calculate C-Space
        if not self._calculate_cspace(robot_radius, obstacle_threshold):
            rospy.logerr("Failed to calculate C-Space. Cannot plan path.")
            return []
        # --- Added C-Space Inspection Logs ---
        if self._cspace_calculated and self.cspace_grid is not None:
            rospy.logwarn(f"Checking calculated cspace_grid. Shape: {self.cspace_grid.shape}")
            rospy.logwarn(f"Number of obstacle cells in cspace_grid: {np.sum(self.cspace_grid)}")
            # Print a sample known to have obstacles (adjust indices based on your map/ROI)
            # Example: Near expected obstacle based on previous logs/viz
            sample_r_start, sample_r_end = 80, 90 # Example row range
            sample_c_start, sample_c_end = 40, 50 # Example col range
            try:
                 # Check if sample indices are valid before slicing
                 if 0 <= sample_r_start < sample_r_end <= self.map_height and \
                    0 <= sample_c_start < sample_c_end <= self.map_width:
                      rospy.logwarn(f"Sample of cspace_grid[{sample_r_start}:{sample_r_end}, {sample_c_start}:{sample_c_end}]:")
                      print(self.cspace_grid[sample_r_start:sample_r_end, sample_c_start:sample_c_end])
                 else:
                      rospy.logwarn("Sample indices are out of map bounds, skipping sample print.")
            except IndexError:
                 rospy.logwarn("IndexError during sample print, skipping.")
        else:
            rospy.logerr("C-Space grid not available for inspection!")
        # --- End C-Space Inspection ---

        rospy.loginfo(f"Planning Boustrophedon path for ROI: {roi} with Robot Radius: {robot_radius}m, Coverage Width: {coverage_width}m")
        waypoints_world = []
        waypoints_grid = [] # Store grid points first

        # 3. Convert ROI to Grid Coordinates
        min_r_corner, min_c_corner = self.world_to_grid(roi['min_x'], roi['min_y'])
        max_r_corner, max_c_corner = self.world_to_grid(roi['max_x'], roi['max_y'])

        if min_r_corner is None or min_c_corner is None or max_r_corner is None or max_c_corner is None:
             rospy.logwarn("ROI corners seem outside map boundaries. Adjusting to map limits might be needed or fail.")
             # A more robust implementation would clip the ROI or find intersection
             rospy.logerr("Cannot determine valid grid range for ROI fully or partially outside map.")
             return []

        r_start = min(min_r_corner, max_r_corner)
        r_end = max(min_r_corner, max_r_corner)
        c_start = min(min_c_corner, max_c_corner)
        c_end = max(min_c_corner, max_c_corner)

        rospy.loginfo(f"Planning for grid ROI: rows {r_start}-{r_end}, cols {c_start}-{c_end}")

        # 4. Calculate Row Step in Grid Units
        effective_coverage_width = coverage_width * (1.0 - overlap_ratio)
        if effective_coverage_width <= 0:
             rospy.logwarn("Effective coverage width <= 0 due to overlap ratio. Using coverage_width directly.")
             effective_coverage_width = coverage_width
        if effective_coverage_width < self.map_resolution:
             rospy.logwarn(f"Effective coverage width ({effective_coverage_width:.3f}m) is less than map resolution ({self.map_resolution:.3f}m). Setting row step to 1 cell.")
             row_step = 1
        else:
             row_step = max(1, int(round(effective_coverage_width / self.map_resolution)))
        rospy.loginfo(f"Calculated Row Step: {row_step} cells for effective coverage width {effective_coverage_width:.3f}m")

        # --- Boustrophedon Logic (Revised Segment Handling + Pre-Append Checks) ---
        direction = 1 # 1: L->R (+c), -1: R->L (-c)
        last_waypoint_cell = None # Track last added GRID cell to avoid duplicates
        current_r = r_start

        rospy.loginfo("Starting Boustrophedon scan loop...")

        while current_r <= r_end:
            # Use logdebug for potentially verbose row processing logs
            rospy.logdebug(f"Processing row {current_r}, direction {direction}")
            segment_start_cell = None # Store the (r, c) of the start of the current segment
            last_free_cell_in_segment = None # Store the last known free cell (r, c) in the current segment

            # Determine column scan range based on direction
            current_c_range = range(c_start, c_end + 1) if direction == 1 else range(c_end, c_start - 1, -1)
            row_has_free_space = False # Track if any valid cell found in this row scan

            # --- Inner loop: Scan columns for the current row ---
            for c in current_c_range:
                # Check if cell is free using the detailed check function
                is_free = self.is_cell_free_in_cspace_roi(current_r, c, roi)

                if is_free:
                    # Current cell is free and valid
                    row_has_free_space = True
                    current_cell = (current_r, c)
                    last_free_cell_in_segment = current_cell # Update the last known free cell

                    if segment_start_cell is None:
                        # This is the beginning of a new segment
                        segment_start_cell = current_cell
                        # rospy.logdebug(f"  Segment started at: {segment_start_cell}") # Can enable for detailed trace

                        # Add the start point, checking for duplicates against the absolute last point added
                        if last_waypoint_cell != segment_start_cell:
                            # --- Pre-Append Check 1 (Start) ---
                            is_still_really_free = self.is_cell_free_in_cspace_roi(segment_start_cell[0], segment_start_cell[1], roi)
                            # rospy.logwarn(f"Waypoint ADD PRE-CHECK (Start): {segment_start_cell}, Check Result: {is_still_really_free}")
                            if not is_still_really_free:
                                 rospy.logerr(f"!!! LOGIC ERROR: Trying to add START point {segment_start_cell} which fails check!")
                            # --- End Pre-Append Check ---
                            waypoints_grid.append(segment_start_cell)
                            last_waypoint_cell = segment_start_cell
                        # else:
                        #      rospy.logdebug(f"  Skipping duplicate segment start waypoint: {segment_start_cell}") # Can enable

                elif segment_start_cell is not None:
                    # Current cell 'c' is NOT free, but we WERE previously in a segment.
                    # This marks the end of the free segment found so far.
                    # rospy.logdebug(f"  Segment ended before cell ({current_r}, {c}) because it's not free.") # Can enable

                    # The actual end of the segment is the 'last_free_cell_in_segment'
                    if last_free_cell_in_segment is not None:
                        # Add the last known free cell as the segment end point, checking for duplicates.
                        # Also ensure the end point is different from the segment start point itself.
                        if last_waypoint_cell != last_free_cell_in_segment and segment_start_cell != last_free_cell_in_segment:
                             # --- Pre-Append Check 2 (End Obstacle) ---
                             is_still_really_free = self.is_cell_free_in_cspace_roi(last_free_cell_in_segment[0], last_free_cell_in_segment[1], roi)
                            #  rospy.logwarn(f"Waypoint ADD PRE-CHECK (End Obstacle): {last_free_cell_in_segment}, Check Result: {is_still_really_free}")
                             if not is_still_really_free:
                                  rospy.logerr(f"!!! LOGIC ERROR: Trying to add END point {last_free_cell_in_segment} which fails check!")
                             # --- End Pre-Append Check ---
                             waypoints_grid.append(last_free_cell_in_segment)
                             last_waypoint_cell = last_free_cell_in_segment
                        # else: # Optional logging for skipped points
                        #     if last_waypoint_cell == last_free_cell_in_segment:
                        #          rospy.logdebug(f"  Skipping duplicate segment end waypoint: {last_free_cell_in_segment}")
                        #     elif segment_start_cell == last_free_cell_in_segment:
                        #          rospy.logdebug(f"  Skipping segment end waypoint because it's same as start: {last_free_cell_in_segment}")

                    # Reset segment tracking for the next potential segment on this row
                    segment_start_cell = None
                    last_free_cell_in_segment = None

            # --- End of inner loop (finished scanning columns for current_r) ---

            # Handle the case where a segment was still active when the row scan finished
            if segment_start_cell is not None:
                 # rospy.logdebug(f"  Segment was active at end of row scan.") # Can enable
                 # The segment ended at the 'last_free_cell_in_segment' found during the scan.
                 if last_free_cell_in_segment is not None:
                     # Add the last known free cell, checking for duplicates and difference from start.
                     if last_waypoint_cell != last_free_cell_in_segment and segment_start_cell != last_free_cell_in_segment:
                          # --- Pre-Append Check 3 (End Boundary) ---
                          is_still_really_free = self.is_cell_free_in_cspace_roi(last_free_cell_in_segment[0], last_free_cell_in_segment[1], roi)
                        #   rospy.logwarn(f"Waypoint ADD PRE-CHECK (End Boundary): {last_free_cell_in_segment}, Check Result: {is_still_really_free}")
                          if not is_still_really_free:
                               rospy.logerr(f"!!! LOGIC ERROR: Trying to add END point {last_free_cell_in_segment} which fails check!")
                          # --- End Pre-Append Check ---
                          waypoints_grid.append(last_free_cell_in_segment)
                          last_waypoint_cell = last_free_cell_in_segment
                     # else: # Optional logging for skipped points
                     #     if last_waypoint_cell == last_free_cell_in_segment:
                     #         rospy.logdebug(f"  Skipping duplicate segment end waypoint at boundary: {last_free_cell_in_segment}")
                     #     elif segment_start_cell == last_free_cell_in_segment:
                     #          rospy.logdebug(f"  Skipping segment end waypoint at boundary because it's same as start: {last_free_cell_in_segment}")


            # Switch direction only if free space was found on this row scan
            if row_has_free_space:
                direction *= -1
                # rospy.logdebug(f"  Row {current_r} had free space. Switched direction to {direction}.") # Can enable
            # else:
            #      rospy.logdebug(f"  Row {current_r} had no free space. Direction remains {direction}.") # Can enable


            # Move to the next row based on step
            current_r += row_step

        rospy.loginfo("Finished Boustrophedon scan loop.")
        # --- Boustrophedon Logic End (Revised + Pre-Append Checks) ---


        # 5. Convert Grid Waypoints to World Coordinates
        last_world_wp = None
        # Filter distance threshold (meters) - avoid points too close together
        # Consider relating this to map_resolution or robot size
        min_dist_threshold = self.map_resolution * 0.25 # Example: 1/4 of cell size

        for r, c in waypoints_grid:
            # Basic check if grid point is valid before conversion
            if not (0 <= r < self.map_height and 0 <= c < self.map_width):
                 rospy.logwarn(f"Skipping invalid grid point ({r},{c}) before world conversion.")
                 continue

            wx, wy = self.grid_to_world(r, c)
            if wx is not None and wy is not None:
                current_world_wp = (wx, wy)
                # Optional: Filter out very close consecutive points
                distance_to_last = math.inf if last_world_wp is None else math.dist(current_world_wp, last_world_wp)

                if distance_to_last > min_dist_threshold:
                    waypoints_world.append(current_world_wp)
                    last_world_wp = current_world_wp
                # else: # Optional log for filtered points
                #     rospy.logdebug(f"Filtered close waypoint: {current_world_wp}, distance: {distance_to_last:.3f}")
            else:
                 rospy.logwarn(f"Failed to convert grid point ({r},{c}) to world coordinates.")


        rospy.loginfo(f"Generated {len(waypoints_world)} world waypoints for ROI (after filtering). Original grid points: {len(waypoints_grid)}")
        if not waypoints_world and (r_end >= r_start and c_end >= c_start):
             rospy.logwarn("No waypoints generated. Check ROI, robot parameters, and C-Space calculation/usage.")
             rospy.logwarn(f"Grid range was r:[{r_start}-{r_end}], c:[{c_start}-{c_end}]. C-Space calculated: {self._cspace_calculated}")

        return waypoints_world

    # ... (Assume publish_cspace_map and other methods exist if added) ...

# Note: Ensure math.dist is available (Python 3.8+) or replace with
# math.sqrt((x1-x2)**2 + (y1-y2)**2) if using older Python.
# Also ensure self.is_cell_free_in_cspace_roi is correctly defined elsewhere in the class.

# --- Main execution block for testing ---
if __name__ == '__main__':
    try:
        rospy.init_node('coverage_planner_tester', anonymous=True)
        planner = CoveragePlanner()

        # Allow time for map processing and potential C-space calculation if needed early
        # Although C-space is calculated lazily in plan_boustrophedon_path now
        rospy.sleep(1.0) # Small delay

        if planner.is_ready():
            # --- Test Parameters ---
            # Define ROI in world coordinates (adjust to your map's coords)
            # Example ROI (might need adjustment based on your map's coordinate system)
            roi_test = {'min_x': 0.0, 'min_y': 0.0, 'max_x': 5.0, 'max_y': 5.0}# Example 4x4m square

            # Define Robot Parameters (adjust to your robot)
            robot_radius_test = 0.105  # meters (for C-Space inflation)
            coverage_width_test = 2 * robot_radius_test # meters (e.g., vacuum width, for row stepping)
            overlap_ratio_test = 0.1  # 10% overlap

            rospy.loginfo("Starting path planning test...")
            waypoints = planner.plan_boustrophedon_path(
                roi=roi_test,
                robot_radius=robot_radius_test, # 用于 C-Space 避障
                coverage_width=coverage_width_test, # 用于计算行间距保证覆盖
                overlap_ratio=overlap_ratio_test
            )

            if waypoints:
                rospy.loginfo("Test Waypoints (World Coordinates):")
                # for i, wp in enumerate(waypoints):
                #     rospy.loginfo(f"  WP {i}: ({wp[0]:.3f}, {wp[1]:.3f})") # Print all points

                # Publish waypoints as a Path for visualization in RViz
                path_pub = rospy.Publisher('/coverage_path', Path, queue_size=1, latch=True)
                path_msg = Path()
                path_msg.header.stamp = rospy.Time.now()
                path_msg.header.frame_id = "map" # Assuming planning is done in map frame
                waypoints_pub=rospy.Publisher("/coverage_waypoints", PoseArray, queue_size=1, latch=True)
                waypoints_msg = PoseArray()
                waypoints_msg.header.stamp = path_msg.header.stamp
                waypoints_msg.header.frame_id = path_msg.header.frame_id
                for i, (wx, wy) in enumerate(waypoints):
                    pose = PoseStamped()
                    pose.header.stamp = path_msg.header.stamp # Use same timestamp for all poses in path
                    pose.header.frame_id = path_msg.header.frame_id
                    pose.pose.position.x = wx
                    pose.pose.position.y = wy
                    pose.pose.orientation.w = 1.0 # Neutral orientation
                    path_msg.poses.append(pose)
                    point=Pose()
                    point.position.x=wx
                    point.position.y=wy
                    point.position.z=0.0
                    if i + 1 < len(waypoints): # 如果存在下一个点
                        next_wx, next_wy = waypoints[i+1]
                        # 计算从当前点到下一个点的角度 (yaw)
                        yaw = math.atan2(next_wy - wy, next_wx - wx)
                    elif i > 0: # 对于最后一个点，使用前一个点的方向
                        prev_wx, prev_wy = waypoints[i-1]
                        yaw = math.atan2(wy - prev_wy, wx - prev_wx)
                    else: # 如果只有一个点，朝向 X 轴正方向 (或任意默认值)
                        yaw = 0.0
                    
                    # 将 yaw 角度转换为四元数 (需要导入 quaternion_from_euler)
                    q = quaternion_from_euler(0, 0, yaw) # roll, pitch, yaw
                    point.orientation.x = q[0]
                    point.orientation.y = q[1]
                    point.orientation.z = q[2]
                    point.orientation.w = q[3]
                    waypoints_msg.poses.append(point)
                waypoints_pub.publish(waypoints_msg)
                path_pub.publish(path_msg)
                rospy.loginfo(f"Published {len(path_msg.poses)} waypoints to /coverage_path for visualization.")
                planner.publish_cspace_map() # Publish C-Space map for visualization
            else:
                rospy.logwarn("Path planning returned no waypoints. Check parameters and map.")

        else:
            rospy.logerr("Planner was not ready (failed to get map?). Exiting.")
        rospy.spin()  # Keep the node alive to listen for messages
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received.")
    except Exception as e:
        rospy.logerr("Unhandled error in planner tester main block.")
        rospy.logerr(traceback.format_exc())
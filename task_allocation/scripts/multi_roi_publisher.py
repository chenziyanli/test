# 文件名: multi_robot_multi_roi_controller.py
#!/usr/bin/env python3
import os
import sys
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose # 导入 Pose
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanRequest # 导入 GetPlan 服务类型
import tf.transformations
import math
import traceback
from functools import partial
import threading

try:

    # --- 假设您的 CoveragePlanner 类现在叫做 'CoveragePlanner' 并且在 'coverage_planner.py' 文件中 ---
    # 您可能需要根据实际情况调整 'from coverage_planner import CoveragePlanner'
    # 检查您的文件名和类名是否匹配
    # from coverage_planner import CoveragePlanner # <--- 确保这里导入正确
    path = os.path.abspath(".")
    sys.path.insert(0,path + "/src/task_allocation/scripts")
    print(sys.path)
    from testplanner import CoveragePlanner
except ImportError as e:
    rospy.logerr(f"Failed to import CoveragePlanner: {e}")
    rospy.logerr("Make sure coverage_planner.py is accessible (check PYTHONPATH, filename, class name).")
    rospy.logerr(f"Current sys.path: {sys.path}")
    exit()
except Exception as e:
    rospy.logerr(f"An unexpected error occurred during CoveragePlanner import: {e}")
    rospy.logerr(traceback.format_exc())
    exit()


# --- 全局变量/配置 ---
robot_namespaces = ['robot1', 'robot2', 'robot3'] # 示例：可以添加更多机器人
# --- 为每个机器人定义 ROI 列表 ---
# robot_rois = {
   
#     'robot1': [
#         {'min_x': -2.0, 'min_y': -2.0, 'max_x': 2.0, 'max_y': 2.0},
#         {'min_x': -2.0, 'min_y': 2.0, 'max_x': 2.0, 'max_y': 5.0},
#         {'min_x': 2.0, 'min_y': 2.0, 'max_x': 5.0, 'max_y': 5.0},
#         # {'min_x': -4.0, 'min_y': -2.0, 'max_x': -2.0, 'max_y': 0.0},
#         # 您可以为 robot1 添加更多 ROI，或为其他 robot 添加列表
#     ],
#     'robot2': [
#         {'min_x': 2.0, 'min_y': -2.0, 'max_x': 5.0, 'max_y': 2.0},
#         {'min_x': 2.0, 'min_y': -5.0, 'max_x': 5.0, 'max_y': -2.0},
#         {'min_x': -2.0, 'min_y': -5.0, 'max_x': 2.0, 'max_y': -2.0},
#     ],
#     'robot3': [
#         {'min_x': -5.0, 'min_y': -5.0, 'max_x': -2.0, 'max_y': -2.0},
#         {'min_x': -5.0, 'min_y': -2.0, 'max_x': -2.0, 'max_y': 2.0},
#         {'min_x': -5.0, 'min_y': 2.0, 'max_x': -2.0, 'max_y': 5.0},

#     ]
# }
robot_rois = {
   
    'robot1': [
        {'min_x': -3.0, 'min_y': -3.0, 'max_x': 3.0, 'max_y': 3.0},
        {'min_x': -5.0, 'min_y': 3.0, 'max_x': 5.0, 'max_y': 5.0},
        {'min_x': -5.0, 'min_y': -5.0, 'max_x': -3.0, 'max_y': 3.0},
        # {'min_x': -3.0, 'min_y': -5.0, 'max_x': 5.0, 'max_y': -3.0}
        # {'min_x': -2.0, 'min_y': -2.0, 'max_x': 2.0, 'max_y': 2.0},
        # {'min_x': -2.0, 'min_y': 2.0, 'max_x': 2.0, 'max_y': 5.0},
        # {'min_x': 2.0, 'min_y': 2.0, 'max_x': 5.0, 'max_y': 5.0},
        # {'min_x': 2.0, 'min_y': -2.0, 'max_x': 5.0, 'max_y': 2.0},
        # {'min_x': 2.0, 'min_y': -5.0, 'max_x': 5.0, 'max_y': -2.0},
        # {'min_x': -2.0, 'min_y': -5.0, 'max_x': 2.0, 'max_y': -2.0},


        # {'min_x': -5.0, 'min_y': -5.0, 'max_x': -2.0, 'max_y': -2.0},
        # {'min_x': -5.0, 'min_y': -2.0, 'max_x': -2.0, 'max_y': 2.0},
        # {'min_x': -5.0, 'min_y': 2.0, 'max_x': -2.0, 'max_y': 5.0},

        # {'min_x': -4.0, 'min_y': -2.0, 'max_x': -2.0, 'max_y': 0.0},
        # 您可以为 robot1 添加更多 ROI，或为其他 robot 添加列表
    ],
    'robot2': [
        {'min_x': 3.0, 'min_y': -5.0, 'max_x': 5.0, 'max_y': 3.2},
        # {'min_x': 2.0, 'min_y': -5.0, 'max_x': 5.0, 'max_y': -2.0},
        # {'min_x': -2.0, 'min_y': -5.0, 'max_x': 2.0, 'max_y': -2.0},
    ],
    'robot3': [
        {'min_x': -3.0, 'min_y': -5.0, 'max_x': 3.0, 'max_y': -3.0},
        # {'min_x': -5.0, 'min_y': -2.0, 'max_x': -2.0, 'max_y': 2.0},
        # {'min_x': -5.0, 'min_y': 2.0, 'max_x': -2.0, 'max_y': 5.0},

    ]
}

clients = {}  # 存储 Action Client
make_plan_services = {} # 存储 make_plan Service Proxy
robot_waypoint_lists = {} # 存储每个机器人的【总】航点列表
robot_waypoint_index = {} # 存储每个机器人下一个要执行的航点索引
path_publishers = {} # 用于存储完整路径 Publisher

# 跟踪机器人状态
robots_ready_for_next = set()
ready_lock = threading.Lock()
robots_finished_all_tasks = set()
finished_lock = threading.Lock()

# --- 新增：用于生成区域间路径的函数 ---
def get_path_between_points(start_point_xy, end_point_xy, robot_ns):
    """
    使用 move_base 的 make_plan 服务规划两个世界坐标点之间的路径。

    Args:
        start_point_xy (tuple): 起始点 (x, y)
        end_point_xy (tuple): 结束点 (x, y)
        robot_ns (str): 机器人的命名空间

    Returns:
        list: 包含路径点 (x, y) 的列表 (不含起点，含终点)，如果规划失败则返回空列表。
    """
    service_name = f'/{robot_ns}/move_base/make_plan'
    if robot_ns not in make_plan_services:
        rospy.loginfo(f"[{robot_ns}] Waiting for make_plan service: {service_name}")
        try:
            rospy.wait_for_service(service_name, timeout=5.0)
            make_plan_services[robot_ns] = rospy.ServiceProxy(service_name, GetPlan)
            rospy.loginfo(f"[{robot_ns}] Connected to make_plan service.")
        except rospy.ROSException as e:
            rospy.logerr(f"[{robot_ns}] Could not connect to make_plan service {service_name}: {e}")
            return []
        except Exception as e:
             rospy.logerr(f"[{robot_ns}] Error creating ServiceProxy for {service_name}: {e}")
             return []

    # 创建 GetPlan 请求
    request = GetPlanRequest()
    request.start.header.frame_id = "map"
    request.start.header.stamp = rospy.Time.now()
    request.start.pose.position.x = start_point_xy[0]
    request.start.pose.position.y = start_point_xy[1]
    request.start.pose.orientation.w = 1.0 # 起始方向通常不重要，设为默认

    request.goal.header.frame_id = "map"
    request.goal.header.stamp = rospy.Time.now()
    request.goal.pose.position.x = end_point_xy[0]
    request.goal.pose.position.y = end_point_xy[1]
    request.goal.pose.orientation.w = 1.0 # 目标方向设为默认

    # 设置容忍度（可选，通常使用 move_base 的默认值）
    request.tolerance = 0.1 # 例如 0.1 米

    try:
        rospy.loginfo(f"[{robot_ns}] Requesting plan from {start_point_xy} to {end_point_xy}")
        response = make_plan_services[robot_ns](request)

        if response.plan.poses: # 检查是否有路径点返回
            rospy.loginfo(f"[{robot_ns}] Received plan with {len(response.plan.poses)} points.")
            # 提取路径点 (x, y)，通常忽略第一个点 (起点)
            path_points = []
            # 跳过第一个点 (request.start)，因为它通常就是上一个段的终点
            for pose_stamped in response.plan.poses[1:]:
                path_points.append((pose_stamped.pose.position.x, pose_stamped.pose.position.y))
            return path_points
        else:
            rospy.logwarn(f"[{robot_ns}] make_plan service returned an empty plan.")
            return []

    except rospy.ServiceException as e:
        rospy.logerr(f"[{robot_ns}] make_plan service call failed: {e}")
        return []
    except Exception as e:
        rospy.logerr(f"[{robot_ns}] Unexpected error during make_plan call: {e}")
        rospy.logerr(traceback.format_exc())
        return []


# --- 回调函数 (与之前版本基本一致) ---
def waypoint_done_callback(status, result, robot_ns):
    """处理航点目标完成事件的回调函数。"""
    global robots_ready_for_next, robots_finished_all_tasks
    status_map = { # Actionlib status codes mapping
        actionlib.GoalStatus.PENDING: 'PENDING', actionlib.GoalStatus.ACTIVE: 'ACTIVE', actionlib.GoalStatus.PREEMPTED: 'PREEMPTED',
        actionlib.GoalStatus.SUCCEEDED: 'SUCCEEDED', actionlib.GoalStatus.ABORTED: 'ABORTED', actionlib.GoalStatus.REJECTED: 'REJECTED',
        actionlib.GoalStatus.PREEMPTING: 'PREEMPTING', actionlib.GoalStatus.RECALLING: 'RECALLING', actionlib.GoalStatus.RECALLED: 'RECALLED',
        actionlib.GoalStatus.LOST: 'LOST'
    }
    status_text = status_map.get(status, 'UNKNOWN')
    # Get the index of the goal that just finished
    current_wp_index = robot_waypoint_index.get(robot_ns, 0) - 1 # Index of the goal that just finished

    rospy.loginfo(f"[{robot_ns}] ***** Callback received for WP {current_wp_index + 1}. Status: {status} ({status_text}) *****")

    if status == actionlib.GoalStatus.SUCCEEDED:
        # 检查是否还有下一个航点
        if current_wp_index + 1 < len(robot_waypoint_lists.get(robot_ns, [])):
            rospy.loginfo(f"[{robot_ns}] Goal succeeded. Marking as ready for next waypoint.")
            with ready_lock:
                robots_ready_for_next.add(robot_ns)
        else:
            rospy.loginfo(f"[{robot_ns}] Final waypoint succeeded. All assigned tasks complete for this robot.")
            with finished_lock:
                 robots_finished_all_tasks.add(robot_ns) # 标记该机器人完成所有任务

    else: # 处理失败或被抢占等情况
        rospy.logwarn(f"[{robot_ns}] Goal did NOT succeed (Status: {status_text}). Stopping tasks for this robot.")
        with finished_lock:
            robots_finished_all_tasks.add(robot_ns) # 也标记为完成（虽然是失败地完成）


# --- 发送下一个航点的函数 (与之前版本基本一致) ---
def send_next_waypoint(robot_ns):
    """为指定的机器人发送其【总】航点列表中的下一个航点。"""
    global robot_waypoint_index

    if robot_ns not in clients:
        rospy.logerr(f"[{robot_ns}] Action client not available in send_next_waypoint.")
        return
    client = clients[robot_ns]
    waypoints = robot_waypoint_lists.get(robot_ns, [])
    current_wp_idx = robot_waypoint_index.get(robot_ns, 0)

    if current_wp_idx >= len(waypoints):
        rospy.logwarn(f"[{robot_ns}] send_next_waypoint called but no more waypoints.")
        with finished_lock:
            if robot_ns not in robots_finished_all_tasks:
                 rospy.loginfo(f"[{robot_ns}] Marking as finished in send_next_waypoint (edge case).")
                 robots_finished_all_tasks.add(robot_ns)
        return

    (x, y) = waypoints[current_wp_idx]
    rospy.loginfo(f"[{robot_ns}] Sending waypoint {current_wp_idx + 1}/{len(waypoints)}: ({x:.2f}, {y:.2f})")

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y

    # --- 设置朝向: 指向下一个点 (如果存在) ---
    if current_wp_idx + 1 < len(waypoints):
        next_x, next_y = waypoints[current_wp_idx + 1]
        # Avoid calculating yaw if current and next points are identical
        if abs(next_x - x) > 1e-3 or abs(next_y - y) > 1e-3:
             yaw = math.atan2(next_y - y, next_x - x)
             q = tf.transformations.quaternion_from_euler(0, 0, yaw)
             goal.target_pose.pose.orientation = Quaternion(*q)
        else:
             # Points are too close, keep previous orientation or default
             goal.target_pose.pose.orientation.w = 1.0
    else:
        # Last waypoint, use default orientation
        goal.target_pose.pose.orientation.w = 1.0
    # --- 朝向设置结束 ---

    # 使用 partial 来传递 robot_ns 到回调函数
    done_cb_with_ns = partial(waypoint_done_callback, robot_ns=robot_ns)

    # 发送目标，并注册完成回调和激活回调（可选）
    # active_cb_with_ns = partial(waypoint_active_callback, robot_ns=robot_ns) # 如果需要激活回调
    # feedback_cb_with_ns = partial(waypoint_feedback_callback, robot_ns=robot_ns) # 如果需要反馈回调
    client.send_goal(goal, done_cb=done_cb_with_ns) #, active_cb=active_cb_with_ns, feedback_cb=feedback_cb_with_ns)

    # 更新索引，指向下一个将要发送的航点
    robot_waypoint_index[robot_ns] = current_wp_idx + 1


# --- 主函数 ---
def main():
    rospy.init_node('multi_robot_multi_roi_controller')
    rospy.loginfo("Multi-robot Multi-ROI Controller Started")

    # 1. 初始化路径规划器
    try:
        planner = CoveragePlanner()
        # 等待地图加载完成
        loop_rate = rospy.Rate(1) # 1 Hz
        while not planner.is_ready() and not rospy.is_shutdown():
            rospy.loginfo_once("Waiting for CoveragePlanner to be ready (map received)...")
            try:
                loop_rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS Interrupt received while waiting for planner.")
                return
        if rospy.is_shutdown(): return # Exit if shutdown during wait
        rospy.loginfo("CoveragePlanner is ready.")
    except Exception as e:
         rospy.logerr("Failed to create or ready CoveragePlanner.")
         rospy.logerr(traceback.format_exc())
         return

    # 2. 初始化 Action Clients 和 Service Proxies
    rospy.loginfo("Initializing action clients and service proxies...")
    connected_robots = []
    for ns in robot_namespaces:
        robot_waypoint_index[ns] = 0 # 初始化索引
        # Action Client
        move_base_client_name = f'/{ns}/move_base'
        rospy.loginfo(f"Connecting to {move_base_client_name} action server...")
        client = actionlib.SimpleActionClient(move_base_client_name, MoveBaseAction)
        rospy.loginfo(f"client: {client}")
        if client.wait_for_server(rospy.Duration(5.0)):
            clients[ns] = client
            rospy.loginfo(f"Connected to {move_base_client_name}")
            # Service Proxy (只为连接成功的机器人创建)
            service_name = f'/{ns}/move_base/make_plan'
            rospy.loginfo(f"Checking for make_plan service: {service_name}")
            rospy.loginfo(f"clients: {clients}")
            try:
                # 检查服务是否存在，避免启动时卡住太久
                rospy.wait_for_service(service_name, timeout=1.0)
                make_plan_services[ns] = rospy.ServiceProxy(service_name, GetPlan)
                rospy.loginfo(f"Found make_plan service for {ns}.")
                connected_robots.append(ns) # 只有 Action 和 Service 都可用才算连接成功
            except rospy.ROSException:
                rospy.logwarn(f"make_plan service {service_name} not available within timeout. Skipping {ns} for inter-ROI planning.")
            except Exception as e:
                rospy.logerr(f"Error creating ServiceProxy for {service_name} for {ns}: {e}")

        else:
            rospy.logerr(f"Failed to connect to {move_base_client_name} action server. Skipping robot {ns}.")
    

    if not clients or not connected_robots: # 需要至少有一个完整的机器人
        rospy.logerr("No action servers found or no robots fully connected (Action + Service). Exiting.")
        return

    # --- 创建路径发布者 (为所有连接的机器人) ---
    rospy.loginfo("Creating path publishers...")
    for ns in connected_robots:
        pub_topic = f'/{ns}/combined_coverage_path' # 新的话题名
        path_publishers[ns] = rospy.Publisher(pub_topic, Path, queue_size=1, latch=True)
        rospy.loginfo(f"Publishing combined path for {ns} on topic: {pub_topic}")

    rospy.loginfo("Initialization complete.")
    rospy.sleep(1.0)

    # 3. 为每个机器人生成【组合】航点列表（覆盖 + 区域间）并发布路径
    rospy.loginfo("Generating and publishing combined coverage waypoints...")
    active_robots_with_paths = [] # 记录实际生成了有效总路径的机器人
    # --- 从参数服务器获取或使用默认值 ---
    robot_radius_param = rospy.get_param("~robot_radius", 0.105) # Example default
    coverage_width_param = rospy.get_param("~coverage_width", 2 * robot_radius_param) # Example default
    overlap_ratio_param = rospy.get_param("~overlap_ratio", 0.1) # Example default
    min_waypoint_distance = rospy.get_param("~min_waypoint_distance", 0.05) # Min dist threshold

    for ns in connected_robots:
        assigned_rois = robot_rois.get(ns, [])
        rospy.loginfo(f"[{ns}] Assigned ROIs: {assigned_rois}")
        if not assigned_rois:
            rospy.logwarn(f"[{ns}] No ROIs assigned. Skipping path generation.")
            continue

        rospy.loginfo(f"[{ns}] Planning combined path for {len(assigned_rois)} ROIs...")
        combined_waypoints_for_robot = []
        last_waypoint_of_previous_segment = None # 跟踪上一个段落(覆盖或连接)的终点

        for i, roi in enumerate(assigned_rois):
            rospy.loginfo(f"[{ns}] Planning coverage for ROI {i+1}: {roi}")
            coverage_waypoints = planner.plan_boustrophedon_path(roi,
                robot_radius=robot_radius_param,
                coverage_width=coverage_width_param,
                overlap_ratio=overlap_ratio_param
            )

            if not coverage_waypoints:
                rospy.logwarn(f"[{ns}] Failed to generate coverage waypoints for ROI {i+1}. Skipping this ROI.")
                continue # 跳过这个ROI

            rospy.loginfo(f"[{ns}] Generated {len(coverage_waypoints)} coverage waypoints for ROI {i+1}.")
            first_waypoint_of_current_coverage = coverage_waypoints[0]

            # --- 生成并添加连接路径 (如果不是第一个 ROI 且 make_plan 服务可用) ---
            if last_waypoint_of_previous_segment is not None and ns in make_plan_services:
                rospy.loginfo(f"[{ns}] Attempting to generate connecting path...")
                inter_roi_path_points = get_path_between_points(
                    last_waypoint_of_previous_segment,
                    first_waypoint_of_current_coverage,
                    ns
                )

                if inter_roi_path_points:
                    rospy.loginfo(f"[{ns}] Adding {len(inter_roi_path_points)} connecting waypoints.")
                    combined_waypoints_for_robot.extend(inter_roi_path_points)
                    # 更新“上一个段落”的终点为连接路径的终点
                    last_waypoint_of_previous_segment = inter_roi_path_points[-1]
                else:
                    rospy.logwarn(f"[{ns}] Failed to generate connecting path to ROI {i+1}. Robot might jump or fail.")
                    # 失败处理：可以尝试直接添加下一个覆盖路径的起点，或标记失败
                    # 这里我们选择直接添加下一个起点，让 action client 尝试过去
                    if combined_waypoints_for_robot and combined_waypoints_for_robot[-1] != first_waypoint_of_current_coverage:
                         combined_waypoints_for_robot.append(first_waypoint_of_current_coverage)
                         last_waypoint_of_previous_segment = first_waypoint_of_current_coverage

            elif last_waypoint_of_previous_segment is not None and ns not in make_plan_services:
                 rospy.logwarn(f"[{ns}] make_plan service not available. Jumping directly to next ROI start.")
                 # 服务不可用，直接跳到下一个ROI起点
                 if combined_waypoints_for_robot and combined_waypoints_for_robot[-1] != first_waypoint_of_current_coverage:
                      combined_waypoints_for_robot.append(first_waypoint_of_current_coverage)
                      last_waypoint_of_previous_segment = first_waypoint_of_current_coverage


            # --- 添加当前 ROI 的覆盖路径 ---
            # 如果是第一个ROI，或者上一个点不是这个覆盖路径的起点
            if not combined_waypoints_for_robot or combined_waypoints_for_robot[-1] != first_waypoint_of_current_coverage:
                 # 如果 combined_waypoints 为空，或者最后一个点不等于当前覆盖起点
                 # （避免在成功生成连接路径后重复添加起点）
                 # 先添加起点
                  if not combined_waypoints_for_robot: # 第一个段落
                      combined_waypoints_for_robot.append(first_waypoint_of_current_coverage)
                  # 在其他情况下，如果连接路径失败并添加了起点，这里不再添加
                  # 如果连接路径成功，其终点理论上就是 first_waypoint_of_current_coverage

            # 添加覆盖路径的剩余部分（从第二个点开始，如果存在的话）
            if len(coverage_waypoints) > 1:
                 combined_waypoints_for_robot.extend(coverage_waypoints[1:])

            # 更新“上一个段落”的终点为当前覆盖路径的终点
            last_waypoint_of_previous_segment = coverage_waypoints[-1]
            #--- ROI 处理结束 ---

        # --- 完成机器人 ns 的所有路径规划 ---
        if combined_waypoints_for_robot:
            # 可选：最终路径过滤，去除距离过近的点
            final_filtered_waypoints = []
            if combined_waypoints_for_robot:
                final_filtered_waypoints.append(combined_waypoints_for_robot[0])
                for j in range(1, len(combined_waypoints_for_robot)):
                    # 使用 math.dist (Python 3.8+) 或自己计算距离
                    try:
                         dist = math.dist(combined_waypoints_for_robot[j], combined_waypoints_for_robot[j-1])
                    except AttributeError: # math.dist not available
                         p1 = combined_waypoints_for_robot[j]
                         p0 = combined_waypoints_for_robot[j-1]
                         dist = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

                    if dist > min_waypoint_distance: # 使用参数化的阈值
                        final_filtered_waypoints.append(combined_waypoints_for_robot[j])

            robot_waypoint_lists[ns] = final_filtered_waypoints
            rospy.loginfo(f"[{ns}] Total combined waypoints: {len(combined_waypoints_for_robot)}, Filtered waypoints: {len(final_filtered_waypoints)}")
            active_robots_with_paths.append(ns) # 记录成功生成路径的机器人

            # --- 发布最终的组合路径 ---
            path_msg = Path()
            path_msg.header.stamp = rospy.Time.now()
            path_msg.header.frame_id = "map"
            for k, (wx, wy) in enumerate(robot_waypoint_lists[ns]):
                 pose_stamped = PoseStamped()
                 pose_stamped.header.stamp = path_msg.header.stamp
                 pose_stamped.header.frame_id = path_msg.header.frame_id
                 pose_stamped.pose.position.x = wx
                 pose_stamped.pose.position.y = wy
                 # 计算朝向
                 if k + 1 < len(robot_waypoint_lists[ns]):
                     next_x, next_y = robot_waypoint_lists[ns][k+1]
                     if abs(next_x - wx) > 1e-3 or abs(next_y - wy) > 1e-3:
                          yaw = math.atan2(next_y - wy, next_x - wx)
                          q = tf.transformations.quaternion_from_euler(0, 0, yaw)
                          pose_stamped.pose.orientation = Quaternion(*q)
                     else:
                          pose_stamped.pose.orientation.w = 1.0
                 else:
                     pose_stamped.pose.orientation.w = 1.0
                 path_msg.poses.append(pose_stamped)

            if ns in path_publishers:
                 path_publishers[ns].publish(path_msg)
                 rospy.loginfo(f"[{ns}] Published final combined path ({len(path_msg.poses)} poses).")
            # --- Path 发布结束 ---

        else:
             rospy.logwarn(f"[{ns}] No valid waypoints generated after processing all assigned ROIs.")
             clients.pop(ns, None) # 从客户端字典移除
             make_plan_services.pop(ns, None) # 从服务字典移除
             path_publishers.pop(ns, None) # 从发布者字典移除


    if not clients or not active_robots_with_paths:
         rospy.logerr("No valid combined waypoints generated for any robot. Exiting.")
         return

    # 4. 为所有有航点的机器人启动第一个航点任务
    rospy.loginfo(f"Sending initial waypoints for active robots: {active_robots_with_paths}")
    with ready_lock:
        for ns in active_robots_with_paths:
            robots_ready_for_next.add(ns)

    # 5. 主循环，监控并发送后续航点 (与之前版本基本一致)
    rate = rospy.Rate(5) # 检查频率
    rospy.loginfo("Controller is running. Monitoring goal completions...")

    # 使用 active_robots_with_paths 来计算当前活动的客户端数量
    initial_active_count = len(active_robots_with_paths)

    while not rospy.is_shutdown():
        # 计算当前真正还在运行任务的机器人数量
        with finished_lock:
            # 确保只计算那些开始时有路径的机器人
            finished_count = len(robots_finished_all_tasks.intersection(set(active_robots_with_paths)))
            current_active_count = initial_active_count - finished_count

        if current_active_count <= 0:
            rospy.loginfo("All active robots have finished their tasks.")
            break # 所有任务完成，退出循环

        robots_to_send_next_list = []
        with ready_lock:
            if robots_ready_for_next:
                 robots_to_send_next_list = list(robots_ready_for_next)
                 robots_ready_for_next.clear()

        if robots_to_send_next_list:
            rospy.loginfo(f"Main loop processing ready robots: {robots_to_send_next_list}")
            for robot_ns in robots_to_send_next_list:
                 # 再次检查机器人是否中途完成了所有任务
                 with finished_lock:
                      if robot_ns not in robots_finished_all_tasks:
                           send_next_waypoint(robot_ns)
                      else:
                           rospy.loginfo(f"[{robot_ns}] was ready but already finished all tasks. Skipping send.")

        try:
            rate.sleep()
        except rospy.ROSTimeMovedBackwardsException:
             rospy.logwarn("ROS Time moved backwards, ignoring sleep.")
        except rospy.ROSInterruptException:
             rospy.loginfo("ROS Interrupt received during main loop.")
             break


    rospy.loginfo(f"Controller finished or interrupted.")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received.")
    except Exception as e:
        rospy.logerr("An unexpected error occurred in main controller.")
        rospy.logerr(traceback.format_exc())
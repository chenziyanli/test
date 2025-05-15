# 文件名: multi_robot_controller.py
#!/usr/bin/env python3
import os
import sys
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion,PoseStamped
import tf.transformations
import math
import traceback
from functools import partial
from nav_msgs.msg import Path
import threading # 需要线程锁来安全地访问共享数据

# 导入我们自己的覆盖规划器类
# 假设 coverage_planner.py 与此脚本在同一个包的 scripts 目录下
# 确保你的包的 CMakeLists.txt 和 package.xml 正确设置了 Python 脚本的安装
try:
    # 如果你在同一个包里
    path = os.path.abspath(".")
    sys.path.insert(0,path + "/src/task_allocation/scripts")
    print(sys.path)
    from testplanner import CoveragePlanner
except ImportError:
    # 如果你把它安装成了一个库
    # from your_package_name.coverage_planner import CoveragePlanner
    rospy.logerr("Failed to import CoveragePlanner. Make sure coverage_planner.py is in the same directory or package.")
    exit()


# 全局变量/配置
robot_namespaces = ['robot1', 'robot2', 'robot3']
# 为每个机器人定义 ROI (现在硬编码，将来由分配算法提供)
# 注意：确保这些 ROI 在地图范围内！
robot_rois = {
    'robot1': [
        {'min_x': 0.0, 'min_y': -2.0, 'max_x': 2.0, 'max_y': 0.0}, # ROI 1 for robot1
        {'min_x': -2.0, 'min_y': -2.0, 'max_x': 0.0, 'max_y': 0.0}  # ROI 2 for robot1
    ],
    'robot2': [
        {'min_x': 0.0, 'min_y': 0.0, 'max_x': 2.0, 'max_y': 2.0}   # ROI 1 for robot2
    ],
    'robot3': [
        {'min_x': -2.0, 'min_y': 0.0, 'max_x': 0.0, 'max_y': 2.0}   # ROI 1 for robot3
    ]
    # 将来这个字典可以由您的任务分配算法动态生成
}

clients = {}  # 存储 Action Client
robot_waypoint_lists = {} # 存储每个机器人的航点列表
robot_waypoint_index = {} # 存储每个机器人下一个要执行的航点索引
path_publishers = {} # <--- 新增：用于存储路径 Publisher 的字典

# *** 新增：用于跟踪哪些机器人已准备好接收下一个航点 ***
# 使用线程安全的集合
robots_ready_for_next = set()
ready_lock = threading.Lock() # 用于保护对上面集合的访问

# *** 新增：用于跟踪哪些机器人完成了所有任务 ***
robots_finished_all_tasks = set()
finished_lock = threading.Lock()

# --- 回调函数 ---
# --- 回调函数 ---
def waypoint_done_callback(status, result, robot_ns):
    """处理航点目标完成事件的回调函数。现在只标记机器人已就绪。"""
    global robots_ready_for_next, robots_finished_all_tasks # 声明全局变量
    status_map = { # ... (状态码映射) ...
        actionlib.GoalStatus.PENDING: 'PENDING', actionlib.GoalStatus.ACTIVE: 'ACTIVE', actionlib.GoalStatus.PREEMPTED: 'PREEMPTED',
        actionlib.GoalStatus.SUCCEEDED: 'SUCCEEDED', actionlib.GoalStatus.ABORTED: 'ABORTED', actionlib.GoalStatus.REJECTED: 'REJECTED',
        actionlib.GoalStatus.PREEMPTING: 'PREEMPTING', actionlib.GoalStatus.RECALLING: 'RECALLING', actionlib.GoalStatus.RECALLED: 'RECALLED',
        actionlib.GoalStatus.LOST: 'LOST'
    }
    status_text = status_map.get(status, 'UNKNOWN')
    current_wp_index = robot_waypoint_index.get(robot_ns, 0) - 1

    rospy.loginfo(f"[{robot_ns}] ***** Callback received for WP {current_wp_index + 1}. Status: {status} ({status_text}) *****")

    if status == actionlib.GoalStatus.SUCCEEDED:
        # 检查是否还有下一个航点
        if current_wp_index + 1 < len(robot_waypoint_lists.get(robot_ns, [])):
            rospy.loginfo(f"[{robot_ns}] Goal succeeded. Marking as ready for next waypoint.")
            # *** 关键修改：不再直接调用 send_next_waypoint ***
            # *** 而是将机器人添加到“就绪”集合中 ***
            with ready_lock:
                robots_ready_for_next.add(robot_ns)
        else:
            rospy.loginfo(f"[{robot_ns}] Final waypoint succeeded. Coverage complete for this robot.")
            with finished_lock:
                 robots_finished_all_tasks.add(robot_ns) # 标记该机器人完成所有任务

    else: # 处理失败或被抢占等情况
        rospy.logwarn(f"[{robot_ns}] Goal did NOT succeed (Status: {status_text}). Stopping coverage for this robot.")
        with finished_lock:
            robots_finished_all_tasks.add(robot_ns) # 也标记为完成（虽然是失败地完成）
        # 这里可以添加更复杂的失败处理，例如记录失败的任务点

# --- 发送下一个航点的函数 ---
def send_next_waypoint(robot_ns):
    """为指定的机器人发送其航点列表中的下一个航点。"""
    global robot_waypoint_index # 需要访问和修改全局索引

    # ... (函数内部逻辑与之前完全相同：获取client, waypoints, index, 创建 goal, 计算 orientation, send_goal 并注册回调, 更新 index) ...
    # ... (为简洁起见，省略重复代码，请参考上一个回复中的函数体) ...
    if robot_ns not in clients: return
    client = clients[robot_ns]
    waypoints = robot_waypoint_lists.get(robot_ns, [])
    current_wp_idx = robot_waypoint_index.get(robot_ns, 0)

    if current_wp_idx >= len(waypoints):
        # 这个检查理论上不应该在这里触发，因为回调函数会先检查
        rospy.logwarn(f"[{robot_ns}] send_next_waypoint called but no more waypoints.")
        with finished_lock:
            robots_finished_all_tasks.add(robot_ns) # 确保标记完成
        return

    (x, y) = waypoints[current_wp_idx]
    rospy.loginfo(f"[{robot_ns}] Sending waypoint {current_wp_idx + 1}/{len(waypoints)}: ({x:.2f}, {y:.2f})")

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y

    if current_wp_idx + 1 < len(waypoints):
        next_x, next_y = waypoints[current_wp_idx + 1]
        yaw = math.atan2(next_y - y, next_x - x)
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation = Quaternion(*q)
    else:
        goal.target_pose.pose.orientation.w = 1.0

    done_cb_with_ns = partial(waypoint_done_callback, robot_ns=robot_ns)
    client.send_goal(goal, done_cb=done_cb_with_ns)
    robot_waypoint_index[robot_ns] = current_wp_idx + 1

# --- 主函数 ---
def main():
    rospy.init_node('multi_robot_controller')
    rospy.loginfo("Multi-robot Controller Started")

    # 1. 初始化路径规划器 (需要先等到地图可用)
    try:
        planner = CoveragePlanner()
        if not planner.is_ready():
            rospy.logerr("Coverage planner failed to initialize (map not received?). Exiting.")
            return
    except Exception as e:
         rospy.logerr("Failed to create CoveragePlanner.")
         rospy.logerr(traceback.format_exc())
         return

    # 2. 初始化 Action Clients
    rospy.loginfo("Initializing action clients...")
    connected_robots = []
    for ns in robot_namespaces:
        robot_waypoint_index[ns] = 0 # 初始化索引
        rospy.loginfo(f"Connecting to /{ns}/move_base...")
        client = actionlib.SimpleActionClient(f'/{ns}/move_base', MoveBaseAction)
        if client.wait_for_server(rospy.Duration(5.0)):
            clients[ns] = client
            connected_robots.append(ns)
            rospy.loginfo(f"Connected to /{ns}/move_base")
        else:
            rospy.logerr(f"Failed to connect to /{ns}/move_base action server.")

    if not clients:
        rospy.logerr("No action servers found. Exiting.")
        return
    rospy.loginfo("Pausing briefly before sending initial waypoints...")
    rospy.sleep(2.0) # 尝试加入 1 到 2 秒的延时，给服务器和客户端充分准备时间
     # *** 新增：为每个成功连接的机器人创建 Path Publisher ***
    rospy.loginfo("Creating path publishers...")
    for ns in connected_robots: # 只为成功连接的机器人创建 publisher
        pub_topic = f'/{ns}/coverage_path' # 例如 /robot1/coverage_path
        # 使用 latch=True 确保 RViz 能收到最后发布的消息
        path_publishers[ns] = rospy.Publisher(pub_topic, Path, queue_size=1, latch=True)
        rospy.loginfo(f"Publishing coverage path for {ns} on topic: {pub_topic}")
    # *** Publisher 创建完毕 ***

    rospy.loginfo("Action clients initialized.")
    rospy.sleep(1.0)



     # 3. 为每个机器人生成航点列表并【发布路径】
    rospy.loginfo("Generating and publishing coverage waypoints...")
    active_robots = [] # 记录实际生成了路径的机器人
    robot_radius_test = 0.105  # meters (for C-Space inflation)
    coverage_width_test = 2 * robot_radius_test # meters (e.g., vacuum width, for row stepping)
    overlap_ratio_test = 0.1  # 10% overlap
    for ns in connected_robots:
        if ns in robot_rois:
            roi = robot_rois[ns]
            rospy.loginfo(f"[{ns}] Planning path for ROI: {roi}")
            waypoints = planner.plan_boustrophedon_path(roi,
                robot_radius=robot_radius_test, # 用于 C-Space 避障
                coverage_width=coverage_width_test, # 用于计算行间距保证覆盖
                overlap_ratio=overlap_ratio_test) # [(x1, y1), ...]
            print(f"[{ns}] Waypoints: {waypoints}")
            if waypoints:
                 robot_waypoint_lists[ns] = waypoints
                 rospy.loginfo(f"[{ns}] Generated {len(waypoints)} waypoints.")
                 active_robots.append(ns) # 将该机器人加入活动列表

                 # *** 新增：创建并发布 Path 消息 ***
                 path_msg = Path()
                 path_msg.header.stamp = rospy.Time.now()
                 path_msg.header.frame_id = "map" # 路径是在 map 坐标系下的

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

                 # 发布路径消息
                 if ns in path_publishers:
                     path_publishers[ns].publish(path_msg)
                     rospy.loginfo(f"[{ns}] Published coverage path ({len(path_msg.poses)} poses).")
                 # *** Path 消息发布完毕 ***
            else:
                 rospy.logwarn(f"[{ns}] Failed to generate waypoints for ROI {roi}. Robot will not move.")
                 clients.pop(ns, None) # 从客户端移除，因为它没有路径
        else:
            rospy.logwarn(f"No ROI defined for robot {ns}. It will not be assigned a coverage task.")
            clients.pop(ns, None) # 从客户端移除

    if not clients or not active_robots: # 检查是否还有可执行任务的机器人
         rospy.logerr("No valid waypoints generated for any active robot. Exiting.")
         return
    # 4. 为所有有航点的机器人启动第一个航点任务 (并行启动)
    rospy.loginfo(f"Sending initial waypoints for active robots: {active_robots}")
    with ready_lock: # 初始化时将所有活动机器人都标记为“就绪”以发送第一个航点
        for ns in active_robots:
            robots_ready_for_next.add(ns)

    rate = rospy.Rate(5) # 设置一个检查频率，例如 5 Hz
    rospy.loginfo("Controller is running. Monitoring goal completions...")

    active_clients_count = len(active_robots) # 初始活动机器人数量

    while not rospy.is_shutdown() and active_clients_count > 0:
        robots_to_send_next = [] # 临时列表存储本次循环要处理的机器人
        # 检查哪些机器人准备好了 (线程安全地访问)
        with ready_lock:
            if robots_ready_for_next:
                 # 复制出来处理，避免在迭代时修改集合
                 robots_to_send_next = list(robots_ready_for_next)
                 robots_ready_for_next.clear() # 清空，等待新的完成信号

        # 为准备好的机器人发送下一个航点
        if robots_to_send_next:
            rospy.loginfo(f"Main loop processing ready robots: {robots_to_send_next}")
            for robot_ns in robots_to_send_next:
                send_next_waypoint(robot_ns)

        # 检查是否有更多机器人完成了所有任务
        with finished_lock:
             active_clients_count = len(active_robots) - len(robots_finished_all_tasks)

        try:
            rate.sleep()
        except rospy.ROSTimeMovedBackwardsException:
             rospy.logwarn("ROS Time moved backwards, ignoring sleep.")
        except rospy.ROSInterruptException:
             rospy.loginfo("ROS Interrupt received during main loop.")
             break


    rospy.loginfo(f"Controller finished or interrupted. Active clients left: {active_clients_count}")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received.")
    except Exception as e:
        rospy.logerr("An unexpected error occurred in main controller.")
        rospy.logerr(traceback.format_exc())
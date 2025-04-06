#!/usr/bin/env python3
import rospy
import actionlib
import json
from math import sqrt
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf2_ros
from dynamic_reconfigure.client import Client
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Twist
import time

pi = 3.14159

WAYPOINTS = [
    (10.5, -3.0),
    (10.5, -12.0),
    (10.5, -21.0)
]

DIST_THRESHOLD = 0.5  # 距离判定阈值

class BridgePatrolAndResponder:
    def __init__(self):
        rospy.init_node("bridge_patrol_and_responder")

        # 初始化 move_base client
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("⏳ Waiting for move_base server...")
        self.client.wait_for_server()
        rospy.loginfo("✅ Connected to move_base")

        self.robot_pose = None
        self.bridge_entry = None
        self.bridge_exit = None
        self.bridge_triggered = False
        self.current_goal = None

        self.start_patrol_flag = False
        rospy.Subscriber('/count_over', Bool, self.count_over_callback)

        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber("/bridge_detector/bridge_head", String, self.json_callback)

        self.vel_client = Client("/move_base/TebLocalPlannerROS")
        # self.vel_client = Client("/move_base/TrajectoryPlannerROS")

        self.bridge_unlock_pub = rospy.Publisher("/cmd_open_bridge", Bool, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.crossed_bridge_pub = rospy.Publisher("/crossed_bridge", Bool, queue_size=1)

    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def json_callback(self, msg):
        try:
            data = json.loads(msg.data)
            bridge_head = data.get("bridge_head", {})
            x = bridge_head.get("x")
            y = bridge_head.get("y")
            if x is not None and y is not None:
                self.bridge_entry = (x, y)
                self.bridge_exit = (x - 3, y)
                rospy.loginfo(f"✅ 收到桥头位置: ({x:.2f}, {y:.2f})")
                self.bridge_triggered = True
        except json.JSONDecodeError:
            rospy.logerr("❌ 无法解析 JSON 格式的数据")

    def count_over_callback(self, msg):
        if msg.data:
            rospy.loginfo("✅ /count_over received True — starting patrol.")
            self.start_patrol_flag = True

    def has_reached_goal(self):
        if self.robot_pose is None or self.current_goal is None:
            return False
        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y
        dx = self.robot_pose.position.x - goal_x
        dy = self.robot_pose.position.y - goal_y
        dist = sqrt(dx ** 2 + dy ** 2)
        # rospy.loginfo(f"x = {self.robot_pose.position.x:.2f}, y = {self.robot_pose.position.y:.2f}), dist = {dist:.2f} ")
        if dist < DIST_THRESHOLD :
            rospy.loginfo("reached goal")
            return True
        else :
            # rospy.loginfo("not reach goal yet")
            return False

    def send_goal(self, x, y, yaw=0.0):
        """
        发送目标点，同时设置期望的偏航角（yaw）
        :param x: 目标点x坐标
        :param y: 目标点y坐标
        :param yaw: 期望偏航角，单位为弧度（默认0.0）
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        # 使用tf转换，将yaw转换为四元数
        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        self.current_goal = goal.target_pose
        rospy.loginfo(f"🚩 Navigating to ({x:.2f}, {y:.2f})")
        self.client.send_goal(goal)

    def go_across_bridge(self):
        if not self.bridge_entry or not self.bridge_exit:
            rospy.logwarn("❌ Bridge entry or exit not defined.")
            return
        ex, ey = self.bridge_entry
        t0 = time.time()
        self.send_goal(ex, ey, pi)
        rospy.loginfo("📏 Waiting to reach bridge entry...")
        while not rospy.is_shutdown():
            if self.has_reached_goal():
                rospy.sleep(2)
                break
        rospy.loginfo(f"✅ reached the bridge entry [{time.time() - t0} s].")

        # # 靠近锥筒
        # t1 = time.time()
        # self.send_goal(ex - 2.2, ey, pi)
        # rospy.loginfo("📏 Waiting to reach unlock entry...")
        # while not rospy.is_shutdown():
        #     if self.has_reached_goal():
        #         rospy.sleep(2)
        #         break
        # rospy.loginfo(f"✅ reached the unlock entry [{time.time() - t1} s].")
                
        # 靠近锥筒
        t1 = time.time()
        self.send_goal(ex - 2.2, ey, pi)
        rospy.loginfo("📏 Waiting to reach unlock entry...")
        self.client.wait_for_result()
        rospy.sleep(1)
        rospy.loginfo(f"✅ reached the unlock entry [{time.time() - t1} s].")

        t2 = time.time()
        rospy.loginfo("🔓 Start Unlocking the bridge...")
        self.bridge_unlock_pub.publish(Bool(data=True))
        rospy.sleep(1)
        rospy.loginfo(f"✅✅✅✅✅✅✅✅✅ Unlocked the bridge [ {time.time() - t2} s]")

        # 直接进行加速直线行驶，不使用导航
        from geometry_msgs.msg import Twist  # Ensure Twist is imported

        # 设置直线行驶的参数
        twist_msg = Twist()
        twist_msg.linear.x = 3.0  # 加速直线速度
        twist_msg.angular.z = 0.0

        # 定义跨越桥面的行驶时长（根据桥长进行调整）
        bridge_cross_duration = 2.2  # 单位：秒
        start_time = rospy.Time.now().to_sec()

        rospy.loginfo("🚀 开始直线行驶过桥...")
        while not rospy.is_shutdown() and (rospy.Time.now().to_sec() - start_time < bridge_cross_duration):
            self.cmd_vel_pub.publish(twist_msg)
            rospy.sleep(0.1)

        # 停止机器人
        twist_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(twist_msg)
        rospy.loginfo("✅ Crossed the bridge via direct straight line driving.")

        # 持续发布 /crossed_bridge=True
        rate = rospy.Rate(1)  # 每秒发布一次
        while not rospy.is_shutdown():
            self.crossed_bridge_pub.publish(Bool(data=True))
            rate.sleep()

        # ox, oy = self.bridge_exit
        # self.send_goal(ox - 2, oy, pi)
        # # 桥上提速
        # self.vel_client.update_configuration({
        #     "max_vel_x": 3.0   # 加速直线速度
        # })
        # self.client.wait_for_result()
        # self.vel_client.update_configuration({
        #     "max_vel_x": 1.0
        # })
        # self.send_goal(ox-3, oy, pi)
        # rospy.loginfo("✅ Crossed the bridge.")

        # # 持续发布 /crossed_bridge=True
        # rate = rospy.Rate(1)  # 每秒发布一次
        # while not rospy.is_shutdown():
        #     self.crossed_bridge_pub.publish(Bool(data=True))
        #     rate.sleep()

    def patrol_loop(self):
        rospy.loginfo("❌ ❌ ❌ ❌ ❌ Waiting for /count_over to start patrol...")
        while not rospy.is_shutdown() and not self.start_patrol_flag:
            rospy.sleep(0.5)
        rospy.loginfo("🚗 Starting patrol...")
        
        index = 0
        forward = True
        rate = rospy.Rate(1)
        x, y = WAYPOINTS[index]
        self.send_goal(x, y, pi)
        while not rospy.is_shutdown():
            if self.bridge_triggered:
                self.go_across_bridge()
                rospy.loginfo("🎉 Bridge process finished. Stopping patrol.")
                break
            if self.has_reached_goal():
                index += 1 if forward else -1
                print("index =", index)
                if index >= len(WAYPOINTS) or index < 0:
                    forward = not forward
                    index = max(0, min(len(WAYPOINTS) - 1, index))
                x, y = WAYPOINTS[index]
                self.send_goal(x, y, pi)
            rate.sleep()

if __name__ == '__main__':
    try:
        navigator = BridgePatrolAndResponder()
        navigator.patrol_loop()
    except rospy.ROSInterruptException:
        pass
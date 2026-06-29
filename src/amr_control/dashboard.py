import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import math

class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        self.sub_odom_raw = self.create_subscription(Odometry, '/odom_raw', self.odom_raw_cb, 10)
        self.sub_odom_camera = self.create_subscription(Odometry, '/odom_camera', self.odom_camera_cb, 10)
        self.sub_odom_camera_aligned = self.create_subscription(Odometry, '/odom_camera_aligned', self.odom_camera_aligned_cb, 10)
        self.sub_odom_filtered = self.create_subscription(Odometry, '/odometry/filtered', self.odom_filtered_cb, 10)
        self.sub_desired = self.create_subscription(Point, '/desired_trajectory', self.desired_cb, 10)
        self.sub_state = self.create_subscription(Float32MultiArray, '/robot_state', self.state_cb, 10)
        self.sub_err = self.create_subscription(Point, '/tracking_error', self.err_cb, 10)
        
        self.raw_x_hist = deque(maxlen=50000)
        self.raw_y_hist = deque(maxlen=50000)
        self.camera_x_hist = deque(maxlen=50000)
        self.camera_y_hist = deque(maxlen=50000)
        self.camera_aligned_x_hist = deque(maxlen=50000)
        self.camera_aligned_y_hist = deque(maxlen=50000)
        self.filtered_x_hist = deque(maxlen=50000)
        self.filtered_y_hist = deque(maxlen=50000)
        self.xd_hist = deque(maxlen=50000)
        self.yd_hist = deque(maxlen=50000)
        
        self.time_hist = deque(maxlen=10000)
        self.rpmL_hist = deque(maxlen=10000)
        self.rpmR_hist = deque(maxlen=10000)
        
        self.err_time_hist = deque(maxlen=10000)
        self.ex_hist = deque(maxlen=10000)
        self.ey_hist = deque(maxlen=10000)
        
        self.start_time = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.odom_received = False
        
    def yaw_from_odom(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_raw_cb(self, msg):
        self.raw_x_hist.append(msg.pose.pose.position.x)
        self.raw_y_hist.append(msg.pose.pose.position.y)

    def odom_camera_cb(self, msg):
        self.camera_x_hist.append(msg.pose.pose.position.x)
        self.camera_y_hist.append(msg.pose.pose.position.y)

    def odom_camera_aligned_cb(self, msg):
        self.camera_aligned_x_hist.append(msg.pose.pose.position.x)
        self.camera_aligned_y_hist.append(msg.pose.pose.position.y)

    def odom_filtered_cb(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.yaw_from_odom(msg)
        
        self.filtered_x_hist.append(self.current_x)
        self.filtered_y_hist.append(self.current_y)
        self.odom_received = True

    def desired_cb(self, msg):
        self.xd_hist.append(msg.x)
        self.yd_hist.append(msg.y)
        
    def state_cb(self, msg):
        if self.start_time is None:
            self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        t = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        self.time_hist.append(t)
        # msg.data[0] = rpmL_signed * 10 từ STM32, cần chia 10
        self.rpmL_hist.append(msg.data[0] / 10.0)
        self.rpmR_hist.append(msg.data[1] / 10.0)

    def err_cb(self, msg):
        if self.start_time is None:
            return
        t = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        self.err_time_hist.append(t)
        self.ex_hist.append(msg.x)
        self.ey_hist.append(msg.y)
        
node = None

def spin_thread():
    rclpy.spin(node)

def main():
    global node
    rclpy.init()
    node = DashboardNode()
    
    t = threading.Thread(target=spin_thread, daemon=True)
    t.start()
    
    # Thiết lập giao diện biểu đồ
    fig = plt.figure(figsize=(12, 8))
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        try:
            plt.style.use('seaborn-darkgrid')
        except OSError:
            plt.style.use('ggplot')
    
    # 1. Biểu đồ Quỹ đạo (X-Y)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Trajectory (X - Y)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    line_traj_d, = ax1.plot([], [], 'r--', linewidth=2, label='Desired', alpha=0.7)
    line_raw, = ax1.plot([], [], color='0.45', linestyle=':', linewidth=1.5, label='Wheel odom (/odom_raw)')
    line_camera, = ax1.plot([], [], 'g.', markersize=2.0, alpha=0.35, label='Camera raw')
    line_camera_aligned, = ax1.plot([], [], color='orange', linestyle='-', linewidth=1.5, label='Camera aligned')
    line_filtered, = ax1.plot([], [], 'b-', linewidth=2, label='Filtered actual')
    ax1.legend()
    # Giữ đúng tỉ lệ mét trên X/Y để vòng tròn không bị vẽ thành ellipse.
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. Biểu đồ Sai số (e_x, e_y)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Tracking Errors (ex, ey)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Meters")
    line_ex, = ax2.plot([], [], 'g-', label='e_x (Longitudinal)')
    line_ey, = ax2.plot([], [], 'm-', label='e_y (Lateral)')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # 3. Biểu đồ RPM 2 bánh
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.set_title("Wheel Velocities (RPM)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("RPM")
    line_rpmL, = ax3.plot([], [], 'r-', label='Left RPM', linewidth=1.5)
    line_rpmR, = ax3.plot([], [], 'b-', label='Right RPM', linewidth=1.5, alpha=0.7)
    ax3.legend()
    
    def update(frame):
        # Update Trajectory
        if (
            len(node.raw_x_hist) > 0
            or len(node.camera_x_hist) > 0
            or len(node.camera_aligned_x_hist) > 0
            or len(node.filtered_x_hist) > 0
            or len(node.xd_hist) > 0
        ):
            line_raw.set_data(node.raw_x_hist, node.raw_y_hist)
            line_camera.set_data(node.camera_x_hist, node.camera_y_hist)
            line_camera_aligned.set_data(node.camera_aligned_x_hist, node.camera_aligned_y_hist)
            line_filtered.set_data(node.filtered_x_hist, node.filtered_y_hist)
            line_traj_d.set_data(node.xd_hist, node.yd_hist)

            all_x = (
                list(node.raw_x_hist)
                + list(node.camera_x_hist)
                + list(node.camera_aligned_x_hist)
                + list(node.filtered_x_hist)
                + list(node.xd_hist)
            )
            all_y = (
                list(node.raw_y_hist)
                + list(node.camera_y_hist)
                + list(node.camera_aligned_y_hist)
                + list(node.filtered_y_hist)
                + list(node.yd_hist)
            )
            
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                # Để lấp đầy khoảng trắng và làm đồ thị to nhất có thể, ta phó thác cho Matplotlib tự tính Limits
                if ax1.get_navigate_mode() is None:
                    # Reset data limits và đưa bounding box của trajectory vào
                    ax1.ignore_existing_data_limits = True
                    ax1.update_datalim([[min_x, min_y], [max_x, max_y]])
                    span_x = max(max_x - min_x, 0.1)
                    span_y = max(max_y - min_y, 0.1)
                    span = max(span_x, span_y)
                    cx = 0.5 * (min_x + max_x)
                    cy = 0.5 * (min_y + max_y)
                    margin = 0.08 * span
                    half = 0.5 * span + margin
                    ax1.set_xlim(cx - half, cx + half)
                    ax1.set_ylim(cy - half, cy + half)
            
        # Update Errors
        if len(node.err_time_hist) > 0:
            line_ex.set_data(node.err_time_hist, node.ex_hist)
            line_ey.set_data(node.err_time_hist, node.ey_hist)
            ax2.set_xlim(0, max(10, node.err_time_hist[-1] + 1))
            
            all_errs = list(node.ex_hist) + list(node.ey_hist)
            if all_errs:
                margin = max(abs(max(all_errs)), abs(min(all_errs))) + 0.05
                ax2.set_ylim(-margin, margin)

        # Update RPM
        if len(node.time_hist) > 0:
            line_rpmL.set_data(node.time_hist, node.rpmL_hist)
            line_rpmR.set_data(node.time_hist, node.rpmR_hist)
            ax3.set_xlim(0, max(10, node.time_hist[-1] + 1))
            
            all_rpms = list(node.rpmL_hist) + list(node.rpmR_hist)
            if all_rpms:
                ax3.set_ylim(min(all_rpms) - 10, max(all_rpms) + 10)
                
        return line_raw, line_camera, line_camera_aligned, line_filtered, line_traj_d, line_ex, line_ey, line_rpmL, line_rpmR
        
    ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import math
import sys
import select
import threading
from std_msgs.msg import Bool, String


class BSMCCircle(Node):
    def __init__(self):
        super().__init__('bsmc_circle')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.err_pub = self.create_publisher(Point, '/tracking_error', 10)
        self.desired_pub = self.create_publisher(Point, '/desired_trajectory', 10)
        self.desired_mode_pub = self.create_publisher(String, '/desired_trajectory_mode', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0

        self.odom_received = False
        self.last_odom_time = None
        self.start_time = None

        self.x0 = 0.0
        self.y0 = 0.0
        self.theta0 = 0.0

        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.STARTUP_DELAY = 3.0
        
        # Pause functionality
        self.is_paused = False
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.pause_sub = self.create_subscription(Bool, '/pause_control', self.pause_cb, 10)
        
        self.kb_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.kb_thread.start()

        # Circle trajectory
        self.R = 1
        self.W = 0.25          # Giảm W để robot chạy chậm hơn, dễ bám
        self.VD = self.R * self.W   # = 0.125 m/s

        # Backstepping gains (Kanayama-stable form) - Giảm đi một nửa để mượt hơn
        self.k1 = 0.4    # longitudinal correction
        self.k2 = 1.2    # lateral correction
        self.k3 = 2.0    # heading correction

        # Weak SMC — Bật lại SMC với hệ số nhỏ ổn định
        self.Ks1 = 0.001
        self.Ks2 = 0.002

        # Tăng lớp biên (boundary layer) để giảm chattering ở RPM
        self.phi1 = 0.8
        self.phi2 = 1.5

        # Coupling lateral error to heading sliding surface
        self.c = 1.0

        # Velocity limits
        self.MAX_V = 0.35
        self.MAX_W = 0.6

        # Bảo vệ không cho bánh đảo chiều
        self.L = 0.17          # wheelbase (m)
        self.VL_MIN = 0.0      # Bỏ clamp này tạm thời để bánh có thể dừng nếu cần

        # Deadband — tăng nhẹ để lọc nhiễu encoder
        self.DEADBAND_EX = 0.005
        self.DEADBAND_EY = 0.005
        self.DEADBAND_ETHETA = 0.01

        # Nếu robot càng chạy càng lệch ngang, đổi thành True
        self.INVERT_EY = False

        self.debug_counter = 0

        self.get_logger().info(
            f"BSMC Circle v3 started. "
            f"R={self.R}, W={self.W}, vd={self.VD:.3f} m/s (~{self.VD/(3.14159*0.063/60.0):.0f} RPM)"
        )
        self.get_logger().info(">>> Nhấn 'p' rồi Enter trên terminal này để TẠM DỪNG / CHẠY TIẾP <<<")

    def pause_cb(self, msg):
        self.toggle_pause(msg.data)

    def keyboard_loop(self):
        while rclpy.ok():
            i, o, e = select.select([sys.stdin], [], [], 0.5)
            if i:
                key = sys.stdin.readline().strip().lower()
                if key == 'p':
                    self.toggle_pause(not self.is_paused)

    def toggle_pause(self, state):
        if self.is_paused != state:
            self.is_paused = state
            if self.is_paused:
                self.get_logger().warn(">>> PAUSED! Gửi lệnh dừng robot. Nhấn 'p'+Enter để chạy tiếp. <<<")
            else:
                self.get_logger().info(">>> RESUMED! Tiếp tục bám quỹ đạo. <<<")

    def sat(self, z):
        return max(-1.0, min(1.0, z))

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def euler_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_theta = self.euler_from_quaternion(
            msg.pose.pose.orientation
        )

        now_s = self.get_clock().now().nanoseconds / 1e9

        if not self.odom_received:
            self.start_time = now_s

            self.x0 = self.current_x
            self.y0 = self.current_y
            self.theta0 = self.current_theta

            self.get_logger().info(
                f"Odometry received. Initial pose: "
                f"x0={self.x0:.3f}, y0={self.y0:.3f}, "
                f"theta0={math.degrees(self.theta0):.1f} deg"
            )

        self.odom_received = True
        self.last_odom_time = now_s

    def generate_desired_trajectory(self, t):
        T_ramp = 2.0
        if t < T_ramp:
            s = self.VD * (t ** 2) / (2.0 * T_ramp)
            v_d = self.VD * t / T_ramp
            w_d = self.W * t / T_ramp
        else:
            s = self.VD * (t - T_ramp / 2.0)
            v_d = self.VD
            w_d = self.W

        # Circle in local frame
        ang = s / self.R
        theta_local = ang
        x_local = self.R * math.sin(ang)
        y_local = self.R * (1.0 - math.cos(ang))

        # Rotate trajectory by initial robot heading
        cos0 = math.cos(self.theta0)
        sin0 = math.sin(self.theta0)

        x_d = self.x0 + cos0 * x_local - sin0 * y_local
        y_d = self.y0 + sin0 * x_local + cos0 * y_local
        theta_d = self.normalize_angle(self.theta0 + theta_local)

        return x_d, y_d, theta_d, v_d, w_d

    def control_loop(self):
        if not self.odom_received:
            return

        now_s = self.get_clock().now().nanoseconds / 1e9

        if self.last_odom_time is not None:
            if now_s - self.last_odom_time > 2.0:
                self.get_logger().warn("Odometry timeout. Stopping robot.")
                self.cmd_pub.publish(Twist())
                return

        # Tính toán thời gian Pause
        if self.is_paused:
            if self.pause_start_time is None:
                self.pause_start_time = now_s
            self.cmd_pub.publish(Twist()) # Liên tục gửi lệnh dừng
            return
        else:
            if self.pause_start_time is not None:
                self.total_paused_time += (now_s - self.pause_start_time)
                self.pause_start_time = None

        # Trừ đi khoảng thời gian đã dừng để quỹ đạo ảo không chạy mất
        t = now_s - self.start_time - self.total_paused_time

        if t < self.STARTUP_DELAY:
            self.cmd_pub.publish(Twist())
            return

        t_track = t - self.STARTUP_DELAY

        x_d, y_d, theta_d, v_d, w_d = self.generate_desired_trajectory(t_track)

        desired_msg = Point()
        desired_msg.x = float(x_d)
        desired_msg.y = float(y_d)
        desired_msg.z = float(theta_d)
        self.desired_pub.publish(desired_msg)

        mode_msg = String()
        mode_msg.data = 'actual'
        self.desired_mode_pub.publish(mode_msg)

        dx = x_d - self.current_x
        dy = y_d - self.current_y

        cos_th = math.cos(self.current_theta)
        sin_th = math.sin(self.current_theta)

        e_x = cos_th * dx + sin_th * dy
        e_y = -sin_th * dx + cos_th * dy

        if self.INVERT_EY:
            e_y = -e_y

        e_theta = self.normalize_angle(theta_d - self.current_theta)

        if abs(e_x) < self.DEADBAND_EX:
            e_x = 0.0

        if abs(e_y) < self.DEADBAND_EY:
            e_y = 0.0

        if abs(e_theta) < self.DEADBAND_ETHETA:
            e_theta = 0.0

        s1 = e_x
        s2 = e_theta + self.c * e_y

        sat_s1 = self.sat(s1 / self.phi1)
        sat_s2 = self.sat(s2 / self.phi2)

        v_cmd = (
            v_d * math.cos(e_theta)
            + self.k1 * e_x
            + self.Ks1 * sat_s1
        )

        w_cmd = (
            w_d
            + self.VD * (self.k2 * e_y + self.k3 * math.sin(e_theta))
            + self.Ks2 * sat_s2
        )

        # Clamp v: đảm bảo luôn tiến về phía trước, floor = VD*0.3
        v_cmd = max(self.VD * 0.3, min(self.MAX_V, v_cmd))

        # Clamp w: bảo vệ bánh không đảo chiều (nếu VL_MIN > 0)
        # v_L = v_cmd - w*(L/2) >= VL_MIN  →  w <= (v_cmd - VL_MIN)/(L/2)
        if self.VL_MIN > 0.0:
            w_max_safe = (v_cmd - self.VL_MIN) / (self.L / 2.0)
            w_limit = min(self.MAX_W, max(0.0, w_max_safe))
        else:
            w_limit = self.MAX_W
        w_cmd = max(-w_limit, min(w_limit, w_cmd))

        err_msg = Point()
        err_msg.x = float(e_x)
        err_msg.y = float(e_y)
        err_msg.z = float(e_theta)
        self.err_pub.publish(err_msg)

        cmd_msg = Twist()
        cmd_msg.linear.x = float(v_cmd)
        cmd_msg.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd_msg)

        self.debug_counter += 1
        if self.debug_counter >= 20:
            self.debug_counter = 0
            self.get_logger().info(
                f"t={t_track:.1f}s | "
                f"ex={e_x:+.3f}, ey={e_y:+.3f}, eth={e_theta:+.3f} | "
                f"v={v_cmd:.3f}, w={w_cmd:.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = BSMCCircle()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

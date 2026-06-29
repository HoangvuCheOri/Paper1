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

        # Figure-8 desired trajectory
        self.A = 0.5
        self.B = 0.25
        self.W = 0.10
        self.VD = math.sqrt((self.A * self.W) ** 2 + (2.0 * self.B * self.W) ** 2)

        # Backstepping gains (Kanayama-stable form)
        self.k1 = 0.8    # longitudinal: correction manh hon
        self.k2 = 2.4    # lateral
        self.k3 = 4.0    # heading

        # Weak SMC
        self.Ks1 = 0.002
        self.Ks2 = 0.005

        self.phi1 = 0.45
        self.phi2 = 1.2

        # Coupling lateral error to heading sliding surface
        self.c = 1.0

        # Velocity limits
        self.MAX_V = 0.35
        self.MAX_W = 0.6

        # Bao ve khong cho banh dao chieu
        self.L = 0.17
        self.VL_MIN = 0.0

        # Deadband
        self.DEADBAND_EX = 0.005
        self.DEADBAND_EY = 0.005
        self.DEADBAND_ETHETA = 0.01

        # Neu robot cang chay cang lech ngang, doi thanh True
        self.INVERT_EY = False

        self.debug_counter = 0

        self.get_logger().info(
            f"BSMC Figure-8 trajectory started. "
            f"A={self.A}, B={self.B}, W={self.W}, vd_nom={self.VD:.3f} m/s"
        )
        self.get_logger().info(">>> Nhan 'p' roi Enter de TAM DUNG / CHAY TIEP <<<")

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
                self.get_logger().warn(">>> PAUSED! Gui lenh dung robot. Nhan 'p'+Enter de chay tiep. <<<")
            else:
                self.get_logger().info(">>> RESUMED! Tiep tuc bam quy dao. <<<")

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
            tau = (t ** 2) / (2.0 * T_ramp)
            tau_dot = t / T_ramp
            tau_ddot = 1.0 / T_ramp
        else:
            tau = t - T_ramp / 2.0
            tau_dot = 1.0
            tau_ddot = 0.0

        wt = self.W * tau
        wt2 = 2.0 * wt

        # Figure-8 in local frame (chưa xoay)
        x_raw = self.A * math.sin(wt)
        y_raw = self.B * math.sin(wt2)

        dx_raw = self.A * self.W * math.cos(wt)
        dy_raw = 2.0 * self.B * self.W * math.cos(wt2)
        ddx_raw = -self.A * self.W * self.W * math.sin(wt)
        ddy_raw = -4.0 * self.B * self.W * self.W * math.sin(wt2)

        # Xoay hình số 8 đi một góc -gamma để tiếp tuyến xuất phát trùng với trục X 
        # (tức là trùng với hướng ban đầu của robot, tránh lỗi bẻ lái gắt đầu game)
        gamma = math.atan2(2.0 * self.B, self.A)
        cos_g = math.cos(-gamma)
        sin_g = math.sin(-gamma)

        x_local = x_raw * cos_g - y_raw * sin_g
        y_local = x_raw * sin_g + y_raw * cos_g

        dx_dtau = dx_raw * cos_g - dy_raw * sin_g
        dy_dtau = dx_raw * sin_g + dy_raw * cos_g

        ddx_dtau = ddx_raw * cos_g - ddy_raw * sin_g
        ddy_dtau = ddx_raw * sin_g + ddy_raw * cos_g

        dx_local = dx_dtau * tau_dot
        dy_local = dy_dtau * tau_dot
        ddx_local = ddx_dtau * tau_dot * tau_dot + dx_dtau * tau_ddot
        ddy_local = ddy_dtau * tau_dot * tau_dot + dy_dtau * tau_ddot

        rot = self.theta0
        cos0 = math.cos(rot)
        sin0 = math.sin(rot)

        x_d = self.x0 + cos0 * x_local - sin0 * y_local
        y_d = self.y0 + sin0 * x_local + cos0 * y_local

        dx_d = cos0 * dx_local - sin0 * dy_local
        dy_d = sin0 * dx_local + cos0 * dy_local
        ddx_d = cos0 * ddx_local - sin0 * ddy_local
        ddy_d = sin0 * ddx_local + cos0 * ddy_local

        v_d = math.sqrt(dx_d * dx_d + dy_d * dy_d)
        if v_d < 1e-6:
            theta_d = self.theta0
            w_d = 0.0
        else:
            theta_d = math.atan2(dy_d, dx_d)
            denom = max(dx_d * dx_d + dy_d * dy_d, 1e-4)
            w_d = (dx_d * ddy_d - dy_d * ddx_d) / denom
            w_d = max(-self.MAX_W, min(self.MAX_W, w_d))

        theta_d = self.normalize_angle(theta_d)

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

        # Tinh toan thoi gian Pause
        if self.is_paused:
            if self.pause_start_time is None:
                self.pause_start_time = now_s
            self.cmd_pub.publish(Twist())
            return
        else:
            if self.pause_start_time is not None:
                self.total_paused_time += (now_s - self.pause_start_time)
                self.pause_start_time = None

        # Tru di khoang thoi gian da dung de quy dao ao khong chay mat
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

        # Clamp v: dam bao luon tien ve phia truoc, floor = VD*0.3
        v_cmd = max(self.VD * 0.3, min(self.MAX_V, v_cmd))

        # Clamp w: bao ve banh khong dao chieu neu VL_MIN > 0
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

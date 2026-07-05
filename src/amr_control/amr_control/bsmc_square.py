import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import math
import sys
import select
import threading
import numpy as np
from std_msgs.msg import Bool, String


class BSMCSquare(Node):
    """
    BSMC Controller for Square Trajectory with Rounded Corners.

    Quỹ đạo hình vuông với các góc bo tròn (corner_radius).
    Robot bắt đầu tại một đỉnh, đi ngược chiều kim đồng hồ (quẹo trái).
    Trên cạnh thẳng: v_d = VD, w_d = 0
    Trên cung tròn góc: v_d = VD, w_d = VD / corner_radius
    """

    def __init__(self):
        super().__init__('bsmc_square')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.err_pub = self.create_publisher(Point, '/tracking_error', 10)
        self.desired_pub = self.create_publisher(Point, '/desired_trajectory', 10)
        self.desired_mode_pub = self.create_publisher(String, '/desired_trajectory_mode', 10)

        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('side_length', 1.0)
        self.declare_parameter('corner_radius', 0.08)
        self.declare_parameter('desired_speed', 0.10)
        self.declare_parameter('control_frequency', 25.0)
        self.declare_parameter('startup_delay', 1.0)
        self.declare_parameter('settle_time', 2.0)
        self.declare_parameter('k1', 0.80)
        self.declare_parameter('k2', 6.0)
        self.declare_parameter('k3', 6.0)
        self.declare_parameter('ks1', 0.0)
        self.declare_parameter('ks2', 0.0)
        self.declare_parameter('phi1', 0.5)
        self.declare_parameter('phi2', 0.8)
        self.declare_parameter('max_v', 0.18)
        self.declare_parameter('max_w', 3.0)
        self.declare_parameter('min_v', 0.0)
        self.declare_parameter('invert_ey', False)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.side_length = float(self.get_parameter('side_length').value)
        self.corner_radius = float(self.get_parameter('corner_radius').value)
        self.VD = float(self.get_parameter('desired_speed').value)

        control_frequency = max(1.0, float(self.get_parameter('control_frequency').value))
        self.timer_period = 1.0 / control_frequency
        self.STARTUP_DELAY = float(self.get_parameter('startup_delay').value)
        self.SETTLE_TIME = float(self.get_parameter('settle_time').value)

        self.k1 = float(self.get_parameter('k1').value)
        self.k2 = float(self.get_parameter('k2').value)
        self.k3 = float(self.get_parameter('k3').value)
        self.Ks1 = float(self.get_parameter('ks1').value)
        self.Ks2 = float(self.get_parameter('ks2').value)
        self.phi1 = float(self.get_parameter('phi1').value)
        self.phi2 = float(self.get_parameter('phi2').value)
        self.MAX_V = float(self.get_parameter('max_v').value)
        self.MAX_W = float(self.get_parameter('max_w').value)
        self.MIN_V = float(self.get_parameter('min_v').value)
        self.INVERT_EY = bool(self.get_parameter('invert_ey').value)

        # Validate
        S = self.side_length
        cr = self.corner_radius
        if S <= 0.0:
            self.get_logger().warn("Invalid side_length <= 0. Using 1.0m.")
            self.side_length = 1.0
            S = 1.0
        if cr < 0.01:
            cr = 0.01
        if cr > S / 2.0:
            cr = S / 2.0
            self.get_logger().warn(f"corner_radius clamped to {cr:.3f}m (max = side/2)")
        self.corner_radius = cr

        # Headroom check
        VD_max = self.MAX_V * 0.60
        if self.VD > VD_max:
            self.VD = VD_max
            self.get_logger().warn(
                f"desired_speed clamped to {self.VD:.3f}m/s (60% of max_v={self.MAX_V:.2f})"
            )

        # Build path segments
        self._build_path()

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_v = 0.0
        self.current_w = 0.0

        self.odom_received = False
        self.last_odom_time = None
        self.start_time = None

        self.x0 = 0.0
        self.y0 = 0.0
        self.theta0 = 0.0
        self.tracking_started = False

        # EKF Settling
        self.settling = False
        self.settle_samples_x = []
        self.settle_samples_y = []
        self.settle_samples_theta = []

        # Yaw bias estimator
        # Tắt yaw bias estimator cho hình vuông (chỉ phù hợp quỹ đạo tròn)
        self.yaw_bias_integral = 0.0
        self.yaw_bias_gain = 0.0
        self.yaw_feedforward = 0.0

        self.timer = self.create_timer(self.timer_period, self.control_loop)

        # Pause
        self.is_paused = False
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.pause_sub = self.create_subscription(Bool, '/pause_control', self.pause_cb, 10)

        self.kb_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.kb_thread.start()

        self.c = 1.0
        self.L = 0.17
        self.VL_MIN = 0.0
        self.DEADBAND_EX = 0.0
        self.DEADBAND_EY = 0.0
        self.DEADBAND_ETHETA = 0.0
        self.debug_counter = 0

        self.get_logger().info(
            f"EKF BSMC Square started: side={S:.2f}m, corner_r={cr:.3f}m, "
            f"vd={self.VD:.3f}m/s, perimeter={self.loop_length:.3f}m, "
            f"lap_time={self.loop_length / self.VD:.1f}s, "
            f"max_v={self.MAX_V:.2f}, max_w={self.MAX_W:.2f}"
        )
        self.get_logger().info(">>> Nhấn 'p' rồi Enter để TẠM DỪNG / CHẠY TIẾP <<<")

    # ─── Path builder ───────────────────────────────────────────────
    def _build_path(self):
        """
        Pre-compute the closed-loop square path with rounded corners.

        Path structure (counterclockwise, local frame):
          Straight 0: (cr,0) → (S-cr,0), heading=0
          Arc 0: 90° left turn, center=(S-cr, cr)
          Straight 1: (S,cr) → (S,S-cr), heading=π/2
          Arc 1: 90° left turn, center=(S-cr, S-cr)
          Straight 2: (S-cr,S) → (cr,S), heading=π
          Arc 2: 90° left turn, center=(cr, S-cr)
          Straight 3: (0,S-cr) → (0,cr), heading=3π/2
          Arc 3: 90° left turn, center=(cr, cr)
          → back to (cr, 0)

        Lead-in: (0,0) → (cr,0) for the first traversal.
        """
        S = self.side_length
        cr = self.corner_radius
        straight_len = S - 2.0 * cr
        arc_len = (math.pi / 2.0) * cr

        headings = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
        starts = [(cr, 0.0), (S, cr), (S - cr, S), (0.0, S - cr)]
        centers = [(S - cr, cr), (S - cr, S - cr), (cr, S - cr), (cr, cr)]
        arc_start_angles = [-math.pi / 2.0, 0.0, math.pi / 2.0, math.pi]

        self.path_segments = []
        for i in range(4):
            self.path_segments.append({
                'type': 'straight',
                'length': straight_len,
                'start': starts[i],
                'heading': headings[i],
            })
            self.path_segments.append({
                'type': 'arc',
                'length': arc_len,
                'center': centers[i],
                'radius': cr,
                'start_angle': arc_start_angles[i],
                'start_heading': headings[i],
            })

        self.loop_length = 4.0 * straight_len + 4.0 * arc_len
        self.lead_in_length = cr  # initial straight from (0,0) to (cr,0)

    def _pose_on_loop(self, s_loop):
        """Return (x_local, y_local, heading) given arc-length on the loop."""
        s_rem = s_loop % self.loop_length
        cumul = 0.0
        for seg in self.path_segments:
            if cumul + seg['length'] > s_rem + 1e-9:
                ds = s_rem - cumul
                if seg['type'] == 'straight':
                    h = seg['heading']
                    sx, sy = seg['start']
                    x = sx + ds * math.cos(h)
                    y = sy + ds * math.sin(h)
                    return x, y, h, 0.0  # w_d = 0 on straights
                else:
                    cx, cy = seg['center']
                    r = seg['radius']
                    angle = seg['start_angle'] + ds / r
                    x = cx + r * math.cos(angle)
                    y = cy + r * math.sin(angle)
                    heading = seg['start_heading'] + ds / r
                    w_local = self.VD / r  # counterclockwise
                    return x, y, heading, w_local
            cumul += seg['length']
        # Fallback
        return 0.0, 0.0, 0.0, 0.0

    # ─── Trajectory generator ──────────────────────────────────────
    def generate_desired_trajectory(self, t):
        VD = self.VD
        cr = self.corner_radius

        # Ramp-up
        T_ramp = 2.5
        if t < T_ramp:
            ramp = t / T_ramp
            s = VD * ramp * (t / 2.0)
            v_d = VD * ramp
        else:
            s = VD * (t - T_ramp / 2.0)
            v_d = VD

        # Lead-in: (0,0) → (cr,0)
        if s < cr:
            x_local = s
            y_local = 0.0
            theta_local = 0.0
            w_d = 0.0
        else:
            s_loop = s - cr
            x_local, y_local, theta_local, w_d = self._pose_on_loop(s_loop)

        # Scale w_d by ramp
        if t < T_ramp:
            w_d *= ramp

        # Rotate by initial heading
        cos0 = math.cos(self.theta0)
        sin0 = math.sin(self.theta0)
        x_d = self.x0 + cos0 * x_local - sin0 * y_local
        y_d = self.y0 + sin0 * x_local + cos0 * y_local
        theta_d = self.normalize_angle(self.theta0 + theta_local)

        return x_d, y_d, theta_d, v_d, w_d

    # ─── Utilities ──────────────────────────────────────────────────
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
                self.get_logger().warn(">>> PAUSED! Nhấn 'p'+Enter để chạy tiếp. <<<")
            else:
                self.get_logger().info(">>> RESUMED! <<<")

    def sat(self, z):
        return max(-1.0, min(1.0, z))

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def euler_from_quaternion(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    # ─── Odom callback ──────────────────────────────────────────────
    def odom_callback(self, msg):
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        new_theta = self.euler_from_quaternion(msg.pose.pose.orientation)
        now_s = self.get_clock().now().nanoseconds / 1e9

        if self.odom_received:
            dx = new_x - self.current_x
            dy = new_y - self.current_y
            if math.sqrt(dx*dx + dy*dy) > 1.0:
                self.get_logger().warn(f"REJECTED outlier jump")
                return

        self.current_x = new_x
        self.current_y = new_y
        self.current_theta = new_theta
        self.current_v = float(msg.twist.twist.linear.x)
        self.current_w = float(msg.twist.twist.angular.z)

        if not self.odom_received:
            self.start_time = now_s
            self.get_logger().info(
                f"Odometry received. Waiting {self.STARTUP_DELAY:.1f}s..."
            )
        self.odom_received = True
        self.last_odom_time = now_s

    # ─── EKF settling ───────────────────────────────────────────────
    def _check_ekf_freshness(self, now_s):
        if self.last_odom_time is None:
            return True
        return (now_s - self.last_odom_time) <= (3.0 / 20.0)

    def _settle_ekf(self, now_s):
        t = now_s - self.start_time - self.total_paused_time
        settle_t = t - self.STARTUP_DELAY

        self.settle_samples_x.append(self.current_x)
        self.settle_samples_y.append(self.current_y)
        self.settle_samples_theta.append(self.current_theta)

        if settle_t >= self.SETTLE_TIME:
            x_arr = np.array(self.settle_samples_x)
            y_arr = np.array(self.settle_samples_y)
            theta_arr = np.array(self.settle_samples_theta)

            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)
            dists = np.sqrt((x_arr - x_mean)**2 + (y_arr - y_mean)**2)
            threshold = np.percentile(dists, 80)
            good = dists <= threshold

            self.x0 = float(np.mean(x_arr[good]))
            self.y0 = float(np.mean(y_arr[good]))
            sin_sum = np.sum(np.sin(theta_arr[good]))
            cos_sum = np.sum(np.cos(theta_arr[good]))
            self.theta0 = math.atan2(sin_sum, cos_sum)

            self.tracking_started = True
            self.settling = False

            n_total = len(self.settle_samples_x)
            n_good = int(good.sum())
            self.get_logger().info(
                f"=== TRACKING START! ({n_good}/{n_total} samples) ==="
                f" x0={self.x0:.4f}, y0={self.y0:.4f}, "
                f"theta0={math.degrees(self.theta0):.1f}deg"
            )
            self.settle_samples_x = []
            self.settle_samples_y = []
            self.settle_samples_theta = []
            return True

        desired_msg = Point()
        desired_msg.x = float(self.current_x)
        desired_msg.y = float(self.current_y)
        desired_msg.z = float(self.current_theta)
        self.desired_pub.publish(desired_msg)

        if len(self.settle_samples_x) % 20 == 0:
            x_arr = np.array(self.settle_samples_x)
            y_arr = np.array(self.settle_samples_y)
            self.get_logger().info(
                f"Settling... {settle_t:.1f}/{self.SETTLE_TIME:.0f}s "
                f"({len(self.settle_samples_x)} samples)"
            )
        return False

    # ─── Control loop ───────────────────────────────────────────────
    def control_loop(self):
        if not self.odom_received:
            return

        now_s = self.get_clock().now().nanoseconds / 1e9

        if self.last_odom_time is not None and now_s - self.last_odom_time > 1.0:
            self.get_logger().warn("Odometry timeout >1s. Stopping.")
            self.cmd_pub.publish(Twist())
            return

        if self.is_paused:
            if self.pause_start_time is None:
                self.pause_start_time = now_s
            self.cmd_pub.publish(Twist())
            return
        else:
            if self.pause_start_time is not None:
                self.total_paused_time += (now_s - self.pause_start_time)
                self.pause_start_time = None

        t = now_s - self.start_time - self.total_paused_time

        if t < self.STARTUP_DELAY:
            self.cmd_pub.publish(Twist())
            return

        if not self.tracking_started:
            if not self.settling:
                self.settling = True
                self.settle_samples_x = []
                self.settle_samples_y = []
                self.settle_samples_theta = []
                self.get_logger().info(
                    f"EKF settling ({self.SETTLE_TIME:.0f}s)..."
                )
            self.cmd_pub.publish(Twist())
            self._settle_ekf(now_s)
            return

        t_track = (
            now_s - self.start_time - self.total_paused_time
            - self.STARTUP_DELAY - self.SETTLE_TIME
        )

        x_d, y_d, theta_d, v_d, w_d = self.generate_desired_trajectory(t_track)

        if not self._check_ekf_freshness(now_s):
            cmd_msg = Twist()
            cmd_msg.linear.x = float(v_d)
            cmd_msg.angular.z = float(w_d)
            self.cmd_pub.publish(cmd_msg)
            return

        desired_msg = Point()
        desired_msg.x = float(x_d)
        desired_msg.y = float(y_d)
        desired_msg.z = float(theta_d)
        self.desired_pub.publish(desired_msg)

        mode_msg = String()
        mode_msg.data = 'actual'
        self.desired_mode_pub.publish(mode_msg)

        # Error computation
        dx = x_d - self.current_x
        dy = y_d - self.current_y
        cos_th = math.cos(self.current_theta)
        sin_th = math.sin(self.current_theta)

        e_x = cos_th * dx + sin_th * dy
        e_y = -sin_th * dx + cos_th * dy
        if self.INVERT_EY:
            e_y = -e_y
        e_theta = self.normalize_angle(theta_d - self.current_theta)

        if abs(e_x) < self.DEADBAND_EX: e_x = 0.0
        if abs(e_y) < self.DEADBAND_EY: e_y = 0.0
        if abs(e_theta) < self.DEADBAND_ETHETA: e_theta = 0.0

        # Sliding surfaces
        s1 = e_x
        s2 = e_theta + self.c * e_y
        sat_s1 = self.sat(s1 / self.phi1)
        sat_s2 = self.sat(s2 / self.phi2)

        # V control
        v_cmd = (
            v_d * math.cos(e_theta)
            + self.k1 * e_x
            + self.Ks1 * sat_s1
        )

        # Yaw bias
        self.yaw_bias_integral += e_theta * self.timer_period
        self.yaw_bias_integral = max(-0.5, min(0.5, self.yaw_bias_integral))
        self.yaw_feedforward = self.yaw_bias_gain * self.yaw_bias_integral

        # W control
        w_cmd = (
            w_d
            + v_d * (self.k2 * e_y + self.k3 * math.sin(e_theta))
            + self.Ks2 * sat_s2
            + self.yaw_feedforward
        )

        # Startup boost
        startup_boost = 1.0
        if t_track < 5.0:
            startup_boost = 1.0 + 0.5 * (1.0 - t_track / 5.0)

        v_cmd = max(self.MIN_V, min(self.MAX_V * startup_boost, v_cmd))

        if self.VL_MIN > 0.0:
            w_max_safe = (v_cmd - self.VL_MIN) / (self.L / 2.0)
            w_limit = min(self.MAX_W, max(0.0, w_max_safe))
        else:
            w_limit = self.MAX_W
        w_cmd = max(-w_limit, min(w_limit, w_cmd))

        # Publish
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
            n_laps = (t_track * self.VD) / self.loop_length if self.loop_length > 0 else 0
            self.get_logger().info(
                f"t={t_track:.1f}s lap={n_laps:.2f} | "
                f"ex={e_x:+.3f} ey={e_y:+.3f} eth={math.degrees(e_theta):+.1f}deg | "
                f"v={v_cmd:.3f} w={w_cmd:.3f} | "
                f"des=({x_d:.2f},{y_d:.2f}) cur=({self.current_x:.2f},{self.current_y:.2f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = BSMCSquare()
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

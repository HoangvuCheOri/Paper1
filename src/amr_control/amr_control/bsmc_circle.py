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


class BSMCCircle(Node):
    def __init__(self):
        super().__init__('bsmc_circle')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.err_pub = self.create_publisher(Point, '/tracking_error', 10)
        self.desired_pub = self.create_publisher(Point, '/desired_trajectory', 10)
        self.desired_mode_pub = self.create_publisher(String, '/desired_trajectory_mode', 10)

        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('radius', 1.1)
        self.declare_parameter('angular_speed', 0.20)
        self.declare_parameter('control_frequency', 25.0)
        self.declare_parameter('startup_delay', 1.0)
        self.declare_parameter('settle_time', 2.0)
        self.declare_parameter('k1', 0.40)
        self.declare_parameter('k2', 2.4)
        self.declare_parameter('k3', 3.5)
        self.declare_parameter('ks1', 0.0)
        self.declare_parameter('ks2', 0.0)
        self.declare_parameter('phi1', 1.0)
        self.declare_parameter('phi2', 1.5)
        self.declare_parameter('max_v', 0.18)
        self.declare_parameter('max_w', 0.85)
        self.declare_parameter('min_v', 0.0)
        self.declare_parameter('invert_ey', False)
        self.declare_parameter('radius_feedback_gain', 0.60)
        self.declare_parameter('radius_position_gain', 0.40)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.R = float(self.get_parameter('radius').value)
        self.W = float(self.get_parameter('angular_speed').value)

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
        self.radius_feedback_gain = float(
            self.get_parameter('radius_feedback_gain').value
        )
        self.radius_position_gain = float(
            self.get_parameter('radius_position_gain').value
        )
        if self.R <= 0.0:
            self.get_logger().warn("Invalid radius <= 0. Using radius=0.5m.")
            self.R = 0.5

        # Đảm bảo VD có headroom để correction: dành 50% cho correction
        self.VD = self.R * self.W
        VD_max_usable = self.MAX_V * 0.60  # chỉ dùng 60% max_v cho feedforward
        if self.VD > VD_max_usable:
            requested_w = self.W
            self.VD = VD_max_usable
            self.W = self.VD / self.R
            self.get_logger().warn(
                f"Requested trajectory speed R*W={self.R * requested_w:.3f}m/s "
                f"exceeds 60% max_v={VD_max_usable:.3f}m/s. "
                f"Reduced angular_speed={self.W:.3f}rad/s, "
                f"vd={self.VD:.3f}m/s (headroom for correction: {self.MAX_V - self.VD:.3f}m/s)"
            )
        else:
            self.get_logger().info(
                f"vd={self.VD:.3f}m/s, max_v={self.MAX_V:.2f}, "
                f"headroom={self.MAX_V - self.VD:.3f}m/s "
                f"({(self.MAX_V - self.VD) / self.VD * 100:.0f}%)"
            )

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

        # Anchor — dùng filter tích lũy nhiều mẫu EKF thay vì 1 mẫu
        self.x0 = 0.0
        self.y0 = 0.0
        self.theta0 = 0.0
        self.tracking_started = False

        # EKF Settling phase
        self.settling = False
        self.settle_samples_x = []
        self.settle_samples_y = []
        self.settle_samples_theta = []

        # Yaw bias estimator: phát hiện robot quay thiếu/quay thừa
        self.yaw_bias_integral = 0.0       # Tích phân sai số hướng
        self.yaw_bias_gain = 0.02          # Tốc độ học yaw bias
        self.yaw_feedforward = 0.0         # Giá trị bù w (rad/s)

        self.timer = self.create_timer(self.timer_period, self.control_loop)

        # Pause functionality
        self.is_paused = False
        self.total_paused_time = 0.0
        self.pause_start_time = None
        self.pause_sub = self.create_subscription(Bool, '/pause_control', self.pause_cb, 10)

        self.kb_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.kb_thread.start()

        # Coupling lateral error to heading sliding surface
        self.c = 1.0

        # Bảo vệ không cho bánh đảo chiều
        self.L = 0.17          # wheelbase (m)
        self.VL_MIN = 0.0

        # Deadband = 0
        self.DEADBAND_EX = 0.0
        self.DEADBAND_EY = 0.0
        self.DEADBAND_ETHETA = 0.0

        self.debug_counter = 0

        self.get_logger().info(
            f"EKF BSMC Circle started: odom_topic={self.odom_topic}, "
            f"R={self.R:.2f}m, W={self.W:.2f}rad/s, vd={self.VD:.3f}m/s, "
            f"max_v={self.MAX_V:.2f}, max_w={self.MAX_W:.2f}, "
            f"k1={self.k1:.2f}, k2={self.k2:.2f}, k3={self.k3:.2f}, "
            f"radius_position_gain={self.radius_position_gain:.2f}, "
            f"radius_feedback_gain={self.radius_feedback_gain:.2f}, "
            f"settle_time={self.SETTLE_TIME:.1f}s"
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
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        new_theta = self.euler_from_quaternion(
            msg.pose.pose.orientation
        )

        now_s = self.get_clock().now().nanoseconds / 1e9

        # Outlier rejection - reject sudden jumps > 1.0m (robot không thể di chuyển nhanh thế)
        if self.odom_received:
            dx = new_x - self.current_x
            dy = new_y - self.current_y
            dist_jump = math.sqrt(dx*dx + dy*dy)

            if dist_jump > 1.0:
                self.get_logger().warn(
                    f"REJECTED outlier: jump={dist_jump:.3f}m "
                    f"({self.current_x:.3f},{self.current_y:.3f}) -> ({new_x:.3f},{new_y:.3f})"
                )
                return  # Skip this measurement, keep old value

        self.current_x = new_x
        self.current_y = new_y
        self.current_theta = new_theta
        self.current_v = float(msg.twist.twist.linear.x)
        self.current_w = float(msg.twist.twist.angular.z)

        if not self.odom_received:
            self.start_time = now_s
            self.get_logger().info(
                f"Odometry received. Waiting {self.STARTUP_DELAY:.1f}s for camera to stabilize..."
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

    def generate_desired_trajectory(self, t):
        T_ramp = 2.5  # Tăng nhẹ ramp để giảm quán tính
        if t < T_ramp:
            # Smooth cubic ramp: gia tốc và gia tốc giật đều liên tục
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

    def _compute_center(self):
        """Tính tâm vòng tròn từ anchor."""
        return (
            self.x0 - self.R * math.sin(self.theta0),
            self.y0 + self.R * math.cos(self.theta0),
        )

    def _compute_radial_error(self):
        """Tính sai số bán kính: dương = robot ở ngoài."""
        cx, cy = self._compute_center()
        dist = math.hypot(self.current_x - cx, self.current_y - cy)
        return dist - self.R

    def _check_ekf_freshness(self, now_s):
        """Kiểm tra EKF còn mới không, nếu trễ > 2 chu kỳ thì dùng feedforward."""
        if self.last_odom_time is None:
            return True
        return (now_s - self.last_odom_time) <= (3.0 / 20.0)  # 150ms

    def _settle_ekf(self, now_s):
        """Giai đoạn làm ổn định EKF: thu thập mẫu để lấy anchor chính xác."""
        t = now_s - self.start_time - self.total_paused_time
        settle_t = t - self.STARTUP_DELAY

        # Lưu mẫu
        self.settle_samples_x.append(self.current_x)
        self.settle_samples_y.append(self.current_y)
        self.settle_samples_theta.append(self.current_theta)

        if settle_t >= self.SETTLE_TIME:
            # Tính anchor TRUNG BÌNH từ các mẫu — loại bỏ outlier
            x_arr = np.array(self.settle_samples_x)
            y_arr = np.array(self.settle_samples_y)
            theta_arr = np.array(self.settle_samples_theta)

            # Loại bỏ 20% mẫu xa nhất (outlier rejection)
            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)
            dists = np.sqrt((x_arr - x_mean)**2 + (y_arr - y_mean)**2)
            threshold = np.percentile(dists, 80)
            good = dists <= threshold

            self.x0 = float(np.mean(x_arr[good]))
            self.y0 = float(np.mean(y_arr[good]))
            # Theta: dùng circular mean
            sin_sum = np.sum(np.sin(theta_arr[good]))
            cos_sum = np.sum(np.cos(theta_arr[good]))
            self.theta0 = math.atan2(sin_sum, cos_sum)

            self.tracking_started = True
            self.settling = False

            n_total = len(self.settle_samples_x)
            n_good = int(good.sum())
            self.get_logger().info(
                f"=== TRACKING START! (sau {self.SETTLE_TIME:.0f}s settling) ==="
                f"Anchored pose (avg of {n_good}/{n_total} samples): "
                f"x0={self.x0:.4f}, y0={self.y0:.4f}, "
                f"theta0={math.degrees(self.theta0):.1f} deg, "
                f"sample_spread_xy={float(np.std(x_arr)):.4f}m"
            )

            # Giải phóng bộ nhớ
            self.settle_samples_x = []
            self.settle_samples_y = []
            self.settle_samples_theta = []
            return True

        # Vẽ desired = current trong lúc settling để camera overlay không bị giật
        desired_msg = Point()
        desired_msg.x = float(self.current_x)
        desired_msg.y = float(self.current_y)
        desired_msg.z = float(self.current_theta)
        self.desired_pub.publish(desired_msg)

        # Debug log trong settling
        if len(self.settle_samples_x) % 20 == 0:
            x_arr = np.array(self.settle_samples_x)
            y_arr = np.array(self.settle_samples_y)
            theta_arr = np.array(self.settle_samples_theta)
            self.get_logger().info(
                f"Settling EKF... {settle_t:.1f}/{self.SETTLE_TIME:.0f}s | "
                f"x_std={float(np.std(x_arr)):.4f}m, "
                f"y_std={float(np.std(y_arr)):.4f}m "
                f"({len(self.settle_samples_x)} samples)"
            )
        return False

    def control_loop(self):
        if not self.odom_received:
            return

        now_s = self.get_clock().now().nanoseconds / 1e9

        if self.last_odom_time is not None:
            if now_s - self.last_odom_time > 1.0:
                self.get_logger().warn("Odometry timeout >1s. Stopping robot.")
                self.cmd_pub.publish(Twist())
                return

        # Tính toán thời gian Pause
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

        # === GIAI ĐOẠN 1: STARTUP DELAY (robot đứng yên) ===
        if t < self.STARTUP_DELAY:
            self.cmd_pub.publish(Twist())
            return

        # === GIAI ĐOẠN 2: EKF SETTLING (robot đứng yên, thu thập mẫu) ===
        if not self.tracking_started:
            if not self.settling:
                self.settling = True
                self.settle_samples_x = []
                self.settle_samples_y = []
                self.settle_samples_theta = []
                self.get_logger().info(
                    f"EKF settling phase started ({self.SETTLE_TIME:.0f}s)... "
                    f"Robot sẽ đứng yên để EKF ổn định."
                )
            self.cmd_pub.publish(Twist())
            self._settle_ekf(now_s)
            return

        # === GIAI ĐOẠN 3: TRACKING ===
        t_track = (
            now_s - self.start_time - self.total_paused_time
            - self.STARTUP_DELAY - self.SETTLE_TIME
        )

        x_d, y_d, theta_d, v_d, w_d = self.generate_desired_trajectory(t_track)

        # Fallback nếu EKF trễ
        if not self._check_ekf_freshness(now_s):
            cmd_msg = Twist()
            cmd_msg.linear.x = float(v_d)
            cmd_msg.angular.z = float(w_d)
            self.cmd_pub.publish(cmd_msg)
            return

        # Publish desired trajectory for camera overlay
        desired_msg = Point()
        desired_msg.x = float(x_d)
        desired_msg.y = float(y_d)
        desired_msg.z = float(theta_d)
        self.desired_pub.publish(desired_msg)

        mode_msg = String()
        mode_msg.data = 'actual'
        self.desired_mode_pub.publish(mode_msg)

        # === TÍNH TOÁN LỖI ===
        dx = x_d - self.current_x
        dy = y_d - self.current_y

        cos_th = math.cos(self.current_theta)
        sin_th = math.sin(self.current_theta)

        e_x = cos_th * dx + sin_th * dy
        e_y = -sin_th * dx + cos_th * dy

        if self.INVERT_EY:
            e_y = -e_y

        e_theta = self.normalize_angle(theta_d - self.current_theta)

        # Deadband
        if abs(e_x) < self.DEADBAND_EX: e_x = 0.0
        if abs(e_y) < self.DEADBAND_EY: e_y = 0.0
        if abs(e_theta) < self.DEADBAND_ETHETA: e_theta = 0.0

        # === SLIDING SURFACES ===
        s1 = e_x
        s2 = e_theta + self.c * e_y

        sat_s1 = self.sat(s1 / self.phi1)
        sat_s2 = self.sat(s2 / self.phi2)

        # === ĐIỀU KHIỂN V ===
        v_cmd = (
            v_d * math.cos(e_theta)
            + self.k1 * e_x
            + self.Ks1 * sat_s1
        )

        # === BÙ YAW BIAS HỌC THÍCH NGHI ===
        # Tích phân e_theta để phát hiện robot quay thiếu/quay thừa một cách hệ thống
        self.yaw_bias_integral += e_theta * self.timer_period
        self.yaw_bias_integral = max(-0.5, min(0.5, self.yaw_bias_integral))
        self.yaw_feedforward = self.yaw_bias_gain * self.yaw_bias_integral

        # === ĐIỀU KHIỂN W ===
        w_cmd = (
            w_d
            + v_d * (self.k2 * e_y + self.k3 * math.sin(e_theta))
            + self.Ks2 * sat_s2
            + self.yaw_feedforward  # Bù yaw bias học được
        )

        # === SAI SỐ BÁN KÍNH ===
        radial_error = self._compute_radial_error()

        # Radius_position_gain: kích hoạt NGAY từ đầu (không cần đợi 1/8 vòng)
        # Vì settling đã ổn định anchor, nên radial error có ý nghĩa ngay.
        # Dùng hệ số ramp để tránh giật khi mới bắt đầu.
        radius_ramp = min(1.0, t_track / 3.0)  # ramp lên trong 3 giây
        if self.radius_position_gain > 0.0:
            turn_direction = 1.0 if self.W >= 0.0 else -1.0
            w_cmd += (
                turn_direction
                * self.radius_position_gain
                * radius_ramp
                * radial_error
            )

        # === RADIUS FEEDBACK (dùng actual v/w) ===
        if (
            self.radius_feedback_gain > 0.0
            and abs(self.current_w) > 0.04
            and abs(self.current_v) > 0.02
        ):
            actual_radius = abs(self.current_v / self.current_w)
            radius_error_ratio = (actual_radius - self.R) / max(self.R, 1e-6)
            radius_error_ratio = max(-1.0, min(1.0, radius_error_ratio))
            turn_direction = 1.0 if (w_d if abs(w_d) > 1e-6 else w_cmd) >= 0.0 else -1.0
            w_cmd += (
                turn_direction
                * abs(self.W)
                * self.radius_feedback_gain
                * radius_error_ratio
            )

        # === BOOST VẬN TỐC KHI KHỞI ĐỘNG ===
        # Dùng boost dựa trên |e_x| thực tế: nếu e_x lớn, robot được phép chạy nhanh hơn
        startup_boost = 1.0
        if t_track < 5.0:
            startup_boost = 1.0 + 0.5 * (1.0 - t_track / 5.0)

        # === CLAMP V ===
        v_cmd = max(self.MIN_V, min(self.MAX_V * startup_boost, v_cmd))

        # === CLAMP W (bảo vệ bánh) ===
        if self.VL_MIN > 0.0:
            w_max_safe = (v_cmd - self.VL_MIN) / (self.L / 2.0)
            w_limit = min(self.MAX_W, max(0.0, w_max_safe))
        else:
            w_limit = self.MAX_W
        w_cmd = max(-w_limit, min(w_limit, w_cmd))

        # === PUBLISH ===
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
            actual_radius_val = float('inf')
            if abs(self.current_w) > 1e-4:
                actual_radius_val = abs(self.current_v / self.current_w)
            self.get_logger().info(
                f"t={t_track:.1f}s | "
                f"ex={e_x:+.3f} ey={e_y:+.3f} eth={math.degrees(e_theta):+.1f}deg | "
                f"rad_err={radial_error:+.3f}m | "
                f"v_cmd={v_cmd:.3f} w_cmd={w_cmd:.3f} "
                f"(v={self.current_v:.3f} w={self.current_w:.3f} R_a={actual_radius_val:.3f}) | "
                f"yaw_ff={self.yaw_feedforward:+.4f} | "
                f"des=({x_d:.2f},{y_d:.2f}) cur=({self.current_x:.2f},{self.current_y:.2f})"
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

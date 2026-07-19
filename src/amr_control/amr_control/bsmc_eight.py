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


class BSMCEight(Node):
    """
    BSMC Controller for Figure-8 (Lemniscate) Trajectory Tracking.

    Quỹ đạo hình số 8 được tham số hóa theo thời gian:
        x_local(t) = A * sin(ω * t)
        y_local(t) = A * sin(ω * t) * cos(ω * t) = (A/2) * sin(2 * ω * t)

    Trong đó:
        A = amplitude (bán kính mỗi nửa vòng, tương đương 'radius')
        ω = tốc độ góc (angular_speed)

    Profile phần cứng bắt đầu tại điểm giao của số 8 và xoay quỹ đạo -45 deg.
    Vì tiếp tuyến Lissajous gốc tại điểm giao là +45 deg, phép xoay này làm
    heading mong muốn ban đầu bằng 0 deg. Robot đặt ở (0, 0, 0) vì vậy chạy
    thẳng theo +X ngay, không quay tại chỗ và không dịch điểm bắt đầu khỏi tâm.
    """

    def __init__(self):
        super().__init__('bsmc_eight')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.err_pub = self.create_publisher(Point, '/tracking_error', 10)
        self.desired_pub = self.create_publisher(Point, '/desired_trajectory', 10)
        self.desired_mode_pub = self.create_publisher(String, '/desired_trajectory_mode', 10)

        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('amplitude', 1.0)       # A: bán kính mỗi nửa vòng (m)
        self.declare_parameter('angular_speed', 0.07)   # ω: tốc độ góc (rad/s)
        self.declare_parameter('trajectory_ramp_time', 12.0)
        self.declare_parameter('control_frequency', 25.0)
        self.declare_parameter('startup_delay', 1.0)
        self.declare_parameter('settle_time', 2.0)
        self.declare_parameter('k1', 0.2205844943)
        self.declare_parameter('k2', 6.5)
        self.declare_parameter('k3', 7.0)
        self.declare_parameter('ks1', 0.03683289)
        self.declare_parameter('ks2', 0.1159106176)
        self.declare_parameter('phi1', 1.0)
        self.declare_parameter('phi2', 1.5)
        self.declare_parameter('max_v', 0.18)
        self.declare_parameter('max_w', 0.85)
        self.declare_parameter('min_v', 0.0)
        self.declare_parameter('invert_ey', False)
        self.declare_parameter('yaw_bias_gain', 0.0)
        # Rotate the standard Lissajous by -45 deg so its centre-crossing
        # tangent is aligned with a robot placed at yaw=0.
        self.declare_parameter('path_rotation_deg', -45.0)
        self.declare_parameter('start_phase_deg', 0.0)
        self.declare_parameter('entry_heading_blend_time', 0.0)
        # Operator-selected rolling start: do not rotate in place. For a clean
        # paper run, physically place the robot near the 45-degree tangent.
        self.declare_parameter('initial_align_time', 0.0)
        self.declare_parameter('initial_align_kp', 1.2)
        self.declare_parameter('initial_align_max_w', 0.35)
        self.declare_parameter('initial_align_tolerance_deg', 3.0)
        self.declare_parameter('initial_align_hold_time', 0.30)
        self.declare_parameter('initial_align_timeout', 12.0)
        self.declare_parameter('initial_align_duration_estimate', 8.0)
        self.declare_parameter('w_feedforward_scale', 0.80)
        self.declare_parameter('w_feedforward_scale_negative', 0.55)
        self.declare_parameter('w_feedforward_scale_positive', 0.80)
        self.declare_parameter('negative_yaw_rate_feedback_gain', 0.50)
        self.declare_parameter('positive_yaw_rate_feedback_gain', 0.30)
        self.declare_parameter('feedback_speed_floor', 0.05)
        self.declare_parameter('center_k1', 1.5)
        self.declare_parameter('center_k1_radius', 0.50)
        self.declare_parameter('center_k2', 18.0)
        self.declare_parameter('center_k2_radius', 0.65)
        self.declare_parameter('center_k3', 2.0)
        self.declare_parameter('center_k3_radius', 0.65)
        self.declare_parameter('v_cmd_scale', 1.0)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.A = float(self.get_parameter('amplitude').value)
        self.W = float(self.get_parameter('angular_speed').value)
        self.TRAJECTORY_RAMP_TIME = max(
            0.1, float(self.get_parameter('trajectory_ramp_time').value)
        )

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
        self.yaw_bias_gain = float(self.get_parameter('yaw_bias_gain').value)
        self.PATH_ROTATION_DEG = float(
            self.get_parameter('path_rotation_deg').value
        )
        self.START_PHASE_DEG = float(
            self.get_parameter('start_phase_deg').value
        )
        self.START_PHASE = math.radians(self.START_PHASE_DEG)
        self.ENTRY_HEADING_BLEND_TIME = max(
            0.0, float(self.get_parameter('entry_heading_blend_time').value)
        )
        self.INITIAL_ALIGN_TIME = max(
            0.0, float(self.get_parameter('initial_align_time').value)
        )
        self.INITIAL_ALIGN_KP = max(
            0.0, float(self.get_parameter('initial_align_kp').value)
        )
        self.INITIAL_ALIGN_MAX_W = max(
            0.0, float(self.get_parameter('initial_align_max_w').value)
        )
        self.INITIAL_ALIGN_TOLERANCE = math.radians(max(
            0.1, float(self.get_parameter('initial_align_tolerance_deg').value)
        ))
        self.INITIAL_ALIGN_HOLD_TIME = max(
            0.0, float(self.get_parameter('initial_align_hold_time').value)
        )
        self.INITIAL_ALIGN_TIMEOUT = max(
            self.INITIAL_ALIGN_TIME,
            float(self.get_parameter('initial_align_timeout').value),
        )
        self.INITIAL_ALIGN_DURATION_ESTIMATE = max(
            0.0, float(
                self.get_parameter('initial_align_duration_estimate').value
            )
        )
        self.W_FEEDFORWARD_SCALE = float(
            self.get_parameter('w_feedforward_scale').value
        )
        self.W_FEEDFORWARD_SCALE_NEGATIVE = float(
            self.get_parameter('w_feedforward_scale_negative').value
        )
        self.W_FEEDFORWARD_SCALE_POSITIVE = float(
            self.get_parameter('w_feedforward_scale_positive').value
        )
        self.NEGATIVE_YAW_RATE_FEEDBACK_GAIN = max(
            0.0, float(
                self.get_parameter('negative_yaw_rate_feedback_gain').value
            )
        )
        self.POSITIVE_YAW_RATE_FEEDBACK_GAIN = max(
            0.0, float(
                self.get_parameter('positive_yaw_rate_feedback_gain').value
            )
        )
        self.FEEDBACK_SPEED_FLOOR = max(
            0.0, float(self.get_parameter('feedback_speed_floor').value)
        )
        center_k1 = float(self.get_parameter('center_k1').value)
        self.CENTER_K1 = self.k1 if center_k1 < 0.0 else center_k1
        self.CENTER_K1_RADIUS = max(
            0.01, float(self.get_parameter('center_k1_radius').value)
        )
        center_k2 = float(self.get_parameter('center_k2').value)
        self.CENTER_K2 = self.k2 if center_k2 < 0.0 else center_k2
        self.CENTER_K2_RADIUS = max(
            0.01, float(self.get_parameter('center_k2_radius').value)
        )
        center_k3 = float(self.get_parameter('center_k3').value)
        self.CENTER_K3 = self.k3 if center_k3 < 0.0 else center_k3
        self.CENTER_K3_RADIUS = max(
            0.01, float(self.get_parameter('center_k3_radius').value)
        )
        self.V_CMD_SCALE = max(
            0.5, min(1.5, float(self.get_parameter('v_cmd_scale').value))
        )

        if self.A <= 0.0:
            self.get_logger().warn("Invalid amplitude <= 0. Using amplitude=0.5m.")
            self.A = 0.5

        # Tính vận tốc tối đa trên quỹ đạo hình số 8. Giá trị này không đổi
        # khi chỉ dịch pha bắt đầu.
        # dx/dt = A*ω*cos(ωt), dy/dt = A*ω*cos(2ωt)
        # Vmax toàn quỹ đạo là A*ω*sqrt(2), không nhất thiết nằm tại thời điểm
        # bắt đầu khi start_phase_deg khác 0.
        self.V_MAX_TRAJECTORY = self.A * self.W * math.sqrt(2.0)
        VD_max_usable = self.MAX_V * 0.60
        if self.V_MAX_TRAJECTORY > VD_max_usable:
            requested_w = self.W
            self.W = VD_max_usable / (self.A * math.sqrt(2.0))
            self.V_MAX_TRAJECTORY = self.A * self.W * math.sqrt(2.0)
            self.get_logger().warn(
                f"Max trajectory speed A*W*sqrt(2)={self.A * requested_w * math.sqrt(2.0):.3f}m/s "
                f"exceeds 60% max_v={VD_max_usable:.3f}m/s. "
                f"Reduced angular_speed={self.W:.3f}rad/s, "
                f"v_max_traj={self.V_MAX_TRAJECTORY:.3f}m/s"
            )
        else:
            self.get_logger().info(
                f"v_max_trajectory={self.V_MAX_TRAJECTORY:.3f}m/s, max_v={self.MAX_V:.2f}, "
                f"headroom={self.MAX_V - self.V_MAX_TRAJECTORY:.3f}m/s "
                f"({(self.MAX_V - self.V_MAX_TRAJECTORY) / max(self.V_MAX_TRAJECTORY, 1e-6) * 100:.0f}%)"
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
        self.yaw_bias_integral = 0.0
        self.yaw_feedforward = 0.0

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
        self.initial_alignment_complete = self.INITIAL_ALIGN_TIME <= 0.0
        self.initial_alignment_in_tolerance_since = None
        self.initial_alignment_failed = False
        self.trajectory_start_track_time = 0.0

        # Chu kỳ hình số 8: T = 2*pi / W
        self.T_period = 2.0 * math.pi / self.W

        self.get_logger().info(
            f"EKF BSMC Figure-8 started: odom_topic={self.odom_topic}, "
            f"A={self.A:.2f}m, W={self.W:.2f}rad/s, "
            f"start_phase={self.START_PHASE_DEG:.1f}deg, "
            f"trajectory_ramp={self.TRAJECTORY_RAMP_TIME:.1f}s, "
            f"entry_heading_blend={self.ENTRY_HEADING_BLEND_TIME:.1f}s, "
            f"v_max_traj={self.V_MAX_TRAJECTORY:.3f}m/s, "
            f"max_v={self.MAX_V:.2f}, max_w={self.MAX_W:.2f}, "
            f"k1={self.k1:.2f}, k2={self.k2:.2f}, k3={self.k3:.2f}, "
            f"T_period={self.T_period:.1f}s, "
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

        # Outlier rejection - reject sudden jumps > 1.0m
        if self.odom_received:
            dx = new_x - self.current_x
            dy = new_y - self.current_y
            dist_jump = math.sqrt(dx*dx + dy*dy)

            if dist_jump > 1.0:
                self.get_logger().warn(
                    f"REJECTED outlier: jump={dist_jump:.3f}m "
                    f"({self.current_x:.3f},{self.current_y:.3f}) -> ({new_x:.3f},{new_y:.3f})"
                )
                return

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
        """
        Sinh quỹ đạo hình số 8 (lemniscate) theo thời gian.

        Tham số hóa trong local frame (trước khi xoay theo theta0):
            x_local(t) = A * sin(ω * t)
            y_local(t) = (A / 2) * sin(2 * ω * t)

        Đạo hàm:
            dx/dt = A * ω * cos(ω * t)
            dy/dt = A * ω * cos(2 * ω * t)

        Vận tốc tuyến tính:
            v_d = sqrt((dx/dt)² + (dy/dt)²)

        Hướng (heading):
            theta_d = atan2(dy/dt, dx/dt)

        Vận tốc góc (curvature-based):
            w_d = d(theta_d)/dt
        """
        A = self.A
        W = self.W

        # Ramp-up phase rate, not only the velocity feedforward.  Keeping
        # phase=W*t while scaling dx/dt made desired position and its
        # derivatives inconsistent during startup.
        T_ramp = self.TRAJECTORY_RAMP_TIME
        if t < T_ramp:
            ramp = t / T_ramp
            phase = W * t * t / (2.0 * T_ramp)
            phase_rate = W * ramp
            phase_accel = W / T_ramp
        else:
            ramp = 1.0
            phase = W * (t - T_ramp / 2.0)
            phase_rate = W
            phase_accel = 0.0

        # Start at a configurable point on the same closed curve. Subtracting
        # that point keeps the initial desired position exactly at (x0, y0),
        # rather than moving or rotating the robot before tracking starts.
        curve_phase = self.START_PHASE + phase
        start_x_local = A * math.sin(self.START_PHASE)
        start_y_local = (A / 2.0) * math.sin(2.0 * self.START_PHASE)
        x_local = A * math.sin(curve_phase) - start_x_local
        y_local = (A / 2.0) * math.sin(2.0 * curve_phase) - start_y_local

        # Đạo hàm bậc 1 (vận tốc)
        dx_dt = A * math.cos(curve_phase) * phase_rate
        dy_dt = A * math.cos(2.0 * curve_phase) * phase_rate

        # Đạo hàm bậc 2 (gia tốc) — cần cho tính w_d
        d2x_dt2 = A * (
            -math.sin(curve_phase) * phase_rate**2
            + math.cos(curve_phase) * phase_accel
        )
        d2y_dt2 = A * (
            -2.0 * math.sin(2.0 * curve_phase) * phase_rate**2
            + math.cos(2.0 * curve_phase) * phase_accel
        )

        # Vận tốc tuyến tính
        v_d = math.sqrt(dx_dt**2 + dy_dt**2)

        # Heading desired. At zero speed use the tangent of the selected start
        # phase explicitly; atan2(0, 0) would create a one-sample jump.
        theta_local = (
            math.atan2(dy_dt, dx_dt)
            if v_d > 1e-9 else math.atan2(
                math.cos(2.0 * self.START_PHASE),
                math.cos(self.START_PHASE),
            )
        )
        # Vận tốc góc (từ curvature): w = (dx*d2y - dy*d2x) / v^2
        v_sq = v_d**2
        if v_sq > 1e-12:
            w_d = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / v_sq
        else:
            w_d = 0.0

        # World-fixed orientation: path_rotation_deg=0 gives a horizontal 8.
        path_rotation = math.radians(self.PATH_ROTATION_DEG)
        cos0 = math.cos(path_rotation)
        sin0 = math.sin(path_rotation)

        x_d = self.x0 + cos0 * x_local - sin0 * y_local
        y_d = self.y0 + sin0 * x_local + cos0 * y_local
        theta_d = self.normalize_angle(path_rotation + theta_local)

        return x_d, y_d, theta_d, v_d, w_d

    def path_center(self):
        """Return the geometric centre after phase translation and rotation."""
        path_rotation = math.radians(self.PATH_ROTATION_DEG)
        cos0 = math.cos(path_rotation)
        sin0 = math.sin(path_rotation)
        start_x = self.A * math.sin(self.START_PHASE)
        start_y = (self.A / 2.0) * math.sin(2.0 * self.START_PHASE)
        return (
            self.x0 - cos0 * start_x + sin0 * start_y,
            self.y0 - sin0 * start_x - cos0 * start_y,
        )

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
            if self.INITIAL_ALIGN_TIME <= 0.0:
                initial_tangent = self.normalize_angle(
                    math.radians(self.PATH_ROTATION_DEG)
                    + math.atan2(
                        math.cos(2.0 * self.START_PHASE),
                        math.cos(self.START_PHASE),
                    )
                )
                initial_heading_error = self.normalize_angle(
                    initial_tangent - self.current_theta
                )
                if abs(initial_heading_error) > math.radians(10.0):
                    self.get_logger().warn(
                        "Rolling start enabled with initial heading mismatch "
                        f"{math.degrees(initial_heading_error):+.1f} deg. "
                        "The controller will move immediately, but a clean "
                        "figure-eight start requires placing the robot near "
                        f"{math.degrees(initial_tangent):.1f} deg before launch."
                    )

            # Giải phóng bộ nhớ
            self.settle_samples_x = []
            self.settle_samples_y = []
            self.settle_samples_theta = []
            return True

        # Vẽ desired = current trong lúc settling
        desired_msg = Point()
        desired_msg.x = float(self.current_x)
        desired_msg.y = float(self.current_y)
        desired_msg.z = float(self.current_theta)
        self.desired_pub.publish(desired_msg)

        # Debug log trong settling
        if len(self.settle_samples_x) % 20 == 0:
            x_arr = np.array(self.settle_samples_x)
            y_arr = np.array(self.settle_samples_y)
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

        # Optional alignment follows the tangent at the selected start phase.
        if not self.initial_alignment_complete:
            target_heading = self.normalize_angle(
                math.radians(self.PATH_ROTATION_DEG)
                + math.atan2(
                    math.cos(2.0 * self.START_PHASE),
                    math.cos(self.START_PHASE),
                )
            )
            heading_error = self.normalize_angle(
                target_heading - self.current_theta
            )
            w_cmd = max(
                -self.INITIAL_ALIGN_MAX_W,
                min(
                    self.INITIAL_ALIGN_MAX_W,
                    self.INITIAL_ALIGN_KP * heading_error,
                ),
            )

            desired_msg = Point()
            desired_msg.x = float(self.x0)
            desired_msg.y = float(self.y0)
            desired_msg.z = float(target_heading)
            self.desired_pub.publish(desired_msg)
            mode_msg = String()
            mode_msg.data = 'actual'
            self.desired_mode_pub.publish(mode_msg)

            err_msg = Point()
            err_msg.z = float(heading_error)
            self.err_pub.publish(err_msg)
            cmd_msg = Twist()
            cmd_msg.angular.z = float(w_cmd)
            if abs(heading_error) <= self.INITIAL_ALIGN_TOLERANCE:
                if self.initial_alignment_in_tolerance_since is None:
                    self.initial_alignment_in_tolerance_since = t_track
                held = t_track - self.initial_alignment_in_tolerance_since
                if (
                    t_track >= self.INITIAL_ALIGN_TIME
                    and held >= self.INITIAL_ALIGN_HOLD_TIME
                ):
                    # Rotation about the wheel axle moves an offset AprilTag
                    # and can also shift the EKF position estimate. Start the
                    # geometric path at the post-alignment pose, not at the
                    # stale pre-rotation anchor (5.57 cm shift in hardware).
                    self.x0 = self.current_x
                    self.y0 = self.current_y
                    self.initial_alignment_complete = True
                    self.trajectory_start_track_time = t_track
                    self.get_logger().info(
                        "Initial heading aligned; re-anchored path at "
                        f"({self.x0:.4f}, {self.y0:.4f}) and starting figure-8 phase."
                    )
            else:
                self.initial_alignment_in_tolerance_since = None

            if not self.initial_alignment_complete:
                if t_track >= self.INITIAL_ALIGN_TIMEOUT:
                    if not self.initial_alignment_failed:
                        self.get_logger().error(
                            "Initial heading alignment timed out; holding stop."
                        )
                        self.initial_alignment_failed = True
                    self.cmd_pub.publish(Twist())
                else:
                    self.cmd_pub.publish(cmd_msg)
                return

            self.cmd_pub.publish(Twist())

        trajectory_t = t_track - self.trajectory_start_track_time

        x_d, y_d, theta_d, v_d, w_d = self.generate_desired_trajectory(trajectory_t)

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

        # Keep the geometric reference unchanged, but introduce its 45-deg
        # initial heading gradually. This prevents the large opposite-sign
        # steering transient seen when starting immediately from yaw=0.
        entry_scale = 1.0
        theta_control = theta_d
        if (
            self.ENTRY_HEADING_BLEND_TIME > 0.0
            and trajectory_t < self.ENTRY_HEADING_BLEND_TIME
        ):
            q = max(0.0, trajectory_t / self.ENTRY_HEADING_BLEND_TIME)
            entry_scale = q * q * (3.0 - 2.0 * q)
            initial_heading = math.radians(self.PATH_ROTATION_DEG)
            heading_from_initial = self.normalize_angle(
                theta_d - initial_heading
            )
            theta_control = self.normalize_angle(
                initial_heading + entry_scale * heading_from_initial
            )
        e_theta = self.normalize_angle(theta_control - self.current_theta)

        # Deadband
        if abs(e_x) < self.DEADBAND_EX: e_x = 0.0
        if abs(e_y) < self.DEADBAND_EY: e_y = 0.0
        if abs(e_theta) < self.DEADBAND_ETHETA: e_theta = 0.0

        # === SLIDING SURFACES ===
        s1 = e_x
        s2 = e_theta + self.c * e_y

        sat_s1 = self.sat(s1 / self.phi1)
        sat_s2 = self.sat(s2 / self.phi2)

        # Smooth, centre-local gains correct phase and lateral offset on the
        # zero-curvature crossing without increasing gains on the outer lobes.
        path_center_x, path_center_y = self.path_center()
        distance_from_center = math.hypot(
            x_d - path_center_x, y_d - path_center_y
        )

        # === ĐIỀU KHIỂN V ===
        # The robot can retain a small phase lead at the centre crossing even
        # while the two outer lobes track well.  Blend a separate longitudinal
        # gain only near the geometric centre so correcting that lead does not
        # disturb the curved outer sections.
        center_q = min(1.0, distance_from_center / self.CENTER_K1_RADIUS)
        center_weight = 1.0 - center_q * center_q * (3.0 - 2.0 * center_q)
        k1_control = self.k1 + center_weight * (self.CENTER_K1 - self.k1)
        v_cmd = (
            v_d * math.cos(e_theta)
            + k1_control * e_x
            + self.Ks1 * sat_s1
        )

        # === BÙ YAW BIAS HỌC THÍCH NGHI ===
        self.yaw_bias_integral += e_theta * self.timer_period
        self.yaw_bias_integral = max(-0.5, min(0.5, self.yaw_bias_integral))
        self.yaw_feedforward = self.yaw_bias_gain * self.yaw_bias_integral

        # === ĐIỀU KHIỂN W ===
        # Smooth interpolation of w gains near w_d=0 to avoid discontinuity
        # at the centre crossing where curvature sign changes.
        W_D_BLEND = 0.02  # rad/s band for linear interpolation
        if abs(w_d) < W_D_BLEND:
            # alpha: 0 = fully negative, 1 = fully positive
            alpha = (w_d + W_D_BLEND) / (2.0 * W_D_BLEND)
            alpha = max(0.0, min(1.0, alpha))
            w_ff_neg = self.W_FEEDFORWARD_SCALE_NEGATIVE if self.W_FEEDFORWARD_SCALE_NEGATIVE > 0.0 else self.W_FEEDFORWARD_SCALE
            w_ff_pos = self.W_FEEDFORWARD_SCALE_POSITIVE if self.W_FEEDFORWARD_SCALE_POSITIVE > 0.0 else self.W_FEEDFORWARD_SCALE
            w_ff_scale = (1.0 - alpha) * w_ff_neg + alpha * w_ff_pos
            yaw_rate_gain = (1.0 - alpha) * self.NEGATIVE_YAW_RATE_FEEDBACK_GAIN + alpha * self.POSITIVE_YAW_RATE_FEEDBACK_GAIN
        elif w_d < 0.0:
            w_ff_scale = self.W_FEEDFORWARD_SCALE_NEGATIVE if self.W_FEEDFORWARD_SCALE_NEGATIVE > 0.0 else self.W_FEEDFORWARD_SCALE
            yaw_rate_gain = self.NEGATIVE_YAW_RATE_FEEDBACK_GAIN
        else:
            w_ff_scale = self.W_FEEDFORWARD_SCALE_POSITIVE if self.W_FEEDFORWARD_SCALE_POSITIVE > 0.0 else self.W_FEEDFORWARD_SCALE
            yaw_rate_gain = self.POSITIVE_YAW_RATE_FEEDBACK_GAIN
        center_k2_q = min(1.0, distance_from_center / self.CENTER_K2_RADIUS)
        center_k2_weight = 1.0 - center_k2_q * center_k2_q * (3.0 - 2.0 * center_k2_q)
        k2_control = self.k2 + center_k2_weight * (self.CENTER_K2 - self.k2)
        center_k3_q = min(1.0, distance_from_center / self.CENTER_K3_RADIUS)
        center_k3_weight = 1.0 - center_k3_q * center_k3_q * (3.0 - 2.0 * center_k3_q)
        k3_control = self.k3 + center_k3_weight * (self.CENTER_K3 - self.k3)
        v_feedback = max(v_d, self.FEEDBACK_SPEED_FLOOR)
        w_cmd = (
            entry_scale * w_ff_scale * w_d
            + v_feedback * (k2_control * e_y + k3_control * math.sin(e_theta))
            + self.Ks2 * sat_s2
            + self.yaw_feedforward
        )
        # Yaw-rate tracking feedback — gain now smoothly interpolated
        # near the centre crossing instead of switching abruptly.
        if yaw_rate_gain > 0.0:
            w_cmd += yaw_rate_gain * (w_cmd - self.current_w)

        # === BOOST VẬN TỐC KHI KHỞI ĐỘNG ===
        startup_boost = 1.0
        if trajectory_t < 5.0:
            startup_boost = 1.0 + 0.5 * (1.0 - trajectory_t / 5.0)

        # === SCALE V (compensate hardware velocity gain) ===
        v_cmd = v_cmd * self.V_CMD_SCALE

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
            # Tính số vòng đã đi
            n_laps = trajectory_t / self.T_period if self.T_period > 0 else 0
            self.get_logger().info(
                f"t={trajectory_t:.1f}s lap={n_laps:.2f} | "
                f"ex={e_x:+.3f} ey={e_y:+.3f} eth={math.degrees(e_theta):+.1f}deg | "
                f"v_cmd={v_cmd:.3f} w_cmd={w_cmd:.3f} "
                f"(v={self.current_v:.3f} w={self.current_w:.3f}) | "
                f"yaw_ff={self.yaw_feedforward:+.4f} | "
                f"des=({x_d:.2f},{y_d:.2f}) cur=({self.current_x:.2f},{self.current_y:.2f})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = BSMCEight()

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

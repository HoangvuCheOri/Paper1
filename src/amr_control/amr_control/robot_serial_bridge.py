import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import serial
import threading
import time

class RobotSerialBridge(Node):
    def __init__(self):
        super().__init__('robot_serial_bridge')
        self.ser = None
        self.connected = False
        
        # CẤU HÌNH CỔNG
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('linear_scale', 1.0)
        # STM32 motor command convention is opposite to ROS angular.z.
        # Keep /cmd_vel ROS-standard: positive angular.z means CCW/left.
        # Giá trị gốc cơ khí. KHÔNG hack bù bán kính ở đây —
        # Camera Homography đã đo chính xác, không cần bù thêm.
        self.declare_parameter('angular_scale', -1.0)
        self.declare_parameter('max_linear_cmd', 0.25)
        self.declare_parameter('max_angular_cmd', 0.90)
        self.declare_parameter('cmd_timeout', 0.25)
        self.declare_parameter('extended_data_order', 'timestamp_seq')
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.linear_scale = float(self.get_parameter('linear_scale').value)
        self.angular_scale = float(self.get_parameter('angular_scale').value)
        self.max_linear_cmd = float(self.get_parameter('max_linear_cmd').value)
        self.max_angular_cmd = float(self.get_parameter('max_angular_cmd').value)
        self.cmd_timeout = max(
            0.05, float(self.get_parameter('cmd_timeout').value)
        )
        self.extended_data_order = str(
            self.get_parameter('extended_data_order').value
        )
        self._last_rx_time = None
        self._last_seq = None
        self._last_cmd_time = None
        self._watchdog_stopped = True
        
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.connected = True
            self.get_logger().info(
                f"Đã kết nối dây qua cổng: {port} - Baud: {baud}; "
                f"linear_scale={self.linear_scale:.2f}, angular_scale={self.angular_scale:.2f}"
            )
        except Exception as e:
            self.get_logger().error(f"Không thể mở cổng Serial: {e}")
            return

        # Pub dữ liệu raw cho state_bridge xử lý
        self.state_pub = self.create_publisher(Float32MultiArray, '/robot_state', 10)
        self.link_pub = self.create_publisher(Float32MultiArray, '/espnow_link', 10)
        self.robot_cmd_pub = self.create_publisher(Twist, '/cmd_vel_robot', 10)
        
        # Sub lệnh vận tốc từ controller bám quỹ đạo
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        # Thread đọc Serial để không làm treo ROS
        self.thread = threading.Thread(target=self.read_serial, daemon=True)
        self.thread.start()
        self.watchdog_timer = self.create_timer(0.05, self.watchdog_callback)

    def read_serial(self):
        while rclpy.ok():
            try:
                if self.ser is None or not self.ser.is_open:
                    return
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    # Legacy: DATA,rpmL*10,rpmR*10,gyro*1000
                    # Extended: DATA,robot_ms,seq,rpmL*10,rpmR*10,gyro*1000
                    if line.startswith("DATA,"):
                        parts = line.split(',')
                        parsed = self.parse_data_packet(parts)
                        if parsed is not None:
                            msg = Float32MultiArray()
                            # Đẩy nguyên cục dữ liệu lên, state_bridge sẽ tự chia 10 và 1000
                            msg.data = [
                                parsed['rpm_l_x10'],
                                parsed['rpm_r_x10'],
                                parsed['gyro_z_x1000'],
                            ]
                            self.state_pub.publish(msg)

                            link_msg = Float32MultiArray()
                            link_msg.data = [
                                parsed['rx_time'],
                                parsed['robot_time_ms'],
                                parsed['seq'],
                                parsed['interarrival_ms'],
                                parsed['seq_gap'],
                            ]
                            self.link_pub.publish(link_msg)
            except Exception as e:
                self.get_logger().error(f"Lỗi đọc Serial: {e}")

    def parse_data_packet(self, parts):
        rx_time = time.monotonic()
        interarrival_ms = float('nan')
        if self._last_rx_time is not None:
            interarrival_ms = (rx_time - self._last_rx_time) * 1000.0
        self._last_rx_time = rx_time

        robot_time_ms = float('nan')
        seq = float('nan')
        seq_gap = float('nan')

        try:
            if len(parts) == 4:
                rpm_l_x10 = float(parts[1])
                rpm_r_x10 = float(parts[2])
                gyro_z_x1000 = float(parts[3])
            elif len(parts) >= 6:
                if self.extended_data_order == 'seq_timestamp':
                    seq = float(parts[1])
                    robot_time_ms = float(parts[2])
                else:
                    robot_time_ms = float(parts[1])
                    seq = float(parts[2])
                rpm_l_x10 = float(parts[3])
                rpm_r_x10 = float(parts[4])
                gyro_z_x1000 = float(parts[5])

                if self._last_seq is not None:
                    seq_gap = max(0.0, seq - self._last_seq - 1.0)
                self._last_seq = seq
            else:
                return None
        except ValueError:
            return None

        return {
            'rx_time': rx_time,
            'robot_time_ms': robot_time_ms,
            'seq': seq,
            'interarrival_ms': interarrival_ms,
            'seq_gap': seq_gap,
            'rpm_l_x10': rpm_l_x10,
            'rpm_r_x10': rpm_r_x10,
            'gyro_z_x1000': gyro_z_x1000,
        }

    def cmd_callback(self, msg):
        if self.ser is None or not self.ser.is_open:
            self.get_logger().warn("Serial chưa sẵn sàng, bỏ qua lệnh /cmd_vel.")
            return

        v_cmd = max(
            -self.max_linear_cmd,
            min(self.max_linear_cmd, msg.linear.x * self.linear_scale),
        )
        w_cmd = max(
            -self.max_angular_cmd,
            min(self.max_angular_cmd, msg.angular.z * self.angular_scale),
        )
        self._last_cmd_time = time.monotonic()
        self._watchdog_stopped = False
        self.write_command(v_cmd, w_cmd)

    def write_command(self, v_cmd, w_cmd):
        """Send and publish the command after bridge scaling/clamping."""
        if self.ser is None or not self.ser.is_open:
            return

        # /cmd_vel stays ROS-standard; angular_scale maps it to STM32 convention.
        cmd_str = f"CMD,{v_cmd:.4f},{w_cmd:.4f}\r\n"
        try:
            self.ser.write(cmd_str.encode())
            sent = Twist()
            sent.linear.x = float(v_cmd)
            sent.angular.z = float(w_cmd)
            self.robot_cmd_pub.publish(sent)
        except Exception as e:
            self.get_logger().error(f"Lỗi gửi Serial: {e}")

    def watchdog_callback(self):
        if self._watchdog_stopped or self._last_cmd_time is None:
            return
        if time.monotonic() - self._last_cmd_time <= self.cmd_timeout:
            return
        self.write_command(0.0, 0.0)
        self._watchdog_stopped = True
        self.get_logger().warn(
            f"/cmd_vel timeout > {self.cmd_timeout:.2f}s; sent stop command."
        )

def main(args=None):
    rclpy.init(args=args)
    node = RobotSerialBridge()
    if not node.connected:
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser is not None and node.ser.is_open:
            node.write_command(0.0, 0.0)
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

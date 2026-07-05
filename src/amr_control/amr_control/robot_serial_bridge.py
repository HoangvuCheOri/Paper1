import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import serial
import threading

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
        self.declare_parameter('angular_scale', -0.7875)
        self.declare_parameter('max_linear_cmd', 0.25)
        self.declare_parameter('max_angular_cmd', 0.90)
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        self.linear_scale = float(self.get_parameter('linear_scale').value)
        self.angular_scale = float(self.get_parameter('angular_scale').value)
        self.max_linear_cmd = float(self.get_parameter('max_linear_cmd').value)
        self.max_angular_cmd = float(self.get_parameter('max_angular_cmd').value)
        
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
        
        # Sub lệnh vận tốc từ controller bám quỹ đạo
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        # Thread đọc Serial để không làm treo ROS
        self.thread = threading.Thread(target=self.read_serial, daemon=True)
        self.thread.start()

    def read_serial(self):
        while rclpy.ok():
            try:
                if self.ser is None or not self.ser.is_open:
                    return
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    # STM32 gửi: DATA,rpmL*10,rpmR*10,gyro*1000
                    if line.startswith("DATA,"):
                        parts = line.split(',')
                        if len(parts) == 4:
                            msg = Float32MultiArray()
                            # Đẩy nguyên cục dữ liệu lên, state_bridge sẽ tự chia 10 và 1000
                            msg.data = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.state_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Lỗi đọc Serial: {e}")

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
        # /cmd_vel stays ROS-standard; angular_scale maps it to STM32 convention.
        cmd_str = f"CMD,{v_cmd:.4f},{w_cmd:.4f}\r\n"
        try:
            self.ser.write(cmd_str.encode())
        except Exception as e:
            self.get_logger().error(f"Lỗi gửi Serial: {e}")

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
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import serial
import threading

class RobotSerialBridge(Node):
    def __init__(self):
        super().__init__('robot_serial_bridge')
        
        # CẤU HÌNH CỔNG
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f"Đã kết nối dây qua cổng: {port} - Baud: {baud}")
        except Exception as e:
            self.get_logger().error(f"Không thể mở cổng Serial: {e}")
            return

        # Pub dữ liệu raw cho state_bridge xử lý
        self.state_pub = self.create_publisher(Float32MultiArray, '/robot_state', 10)
        
        # Sub lệnh vận tốc từ bsmc_controller
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        # Thread đọc Serial để không làm treo ROS
        self.thread = threading.Thread(target=self.read_serial, daemon=True)
        self.thread.start()

    def read_serial(self):
        while rclpy.ok():
            if self.ser.in_waiting > 0:
                try:
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
        # Gửi lệnh trực tiếp (không đảo dấu) để robot chạy đúng hướng CCW (quay trái, Y dương)
        cmd_str = f"CMD,{msg.linear.x:.4f},{msg.angular.z:.4f}\r\n"
        try:
            self.ser.write(cmd_str.encode())
        except Exception as e:
            self.get_logger().error(f"Lỗi gửi Serial: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'ser'):
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

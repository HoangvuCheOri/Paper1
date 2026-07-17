#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import threading


class SpinTestNode(Node):
    """
    Node đơn giản để điều khiển robot quay tại chỗ liên tục
    nhằm mục đích test tính ổn định của đường truyền ESP-NOW.
    """
    def __init__(self):
        super().__init__('spin_test')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Các tham số cấu hình
        self.declare_parameter('angular_speed', 0.5)  # rad/s
        self.declare_parameter('hz', 25.0)            # Tần số publish

        self.w = self.get_parameter('angular_speed').value
        hz = max(1.0, self.get_parameter('hz').value)

        self.timer = self.create_timer(1.0 / hz, self.timer_cb)

        self.is_paused = False
        self.kb_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.kb_thread.start()

        self.get_logger().info(f"Spin test started: angular_speed={self.w:.2f} rad/s")
        self.get_logger().info(">>> Nhấn 'p' + Enter để tạm dừng/chạy tiếp. <<<")

    def keyboard_loop(self):
        while rclpy.ok():
            i, o, e = select.select([sys.stdin], [], [], 0.5)
            if i:
                key = sys.stdin.readline().strip().lower()
                if key == 'p':
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        self.get_logger().warn(">>> ĐÃ DỪNG! Lệnh vận tốc = 0 <<<")
                        self.pub.publish(Twist())
                    else:
                        self.get_logger().info(">>> TIẾP TỤC QUAY <<<")

    def timer_cb(self):
        msg = Twist()
        if not self.is_paused:
            msg.angular.z = float(self.w)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpinTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Đảm bảo robot dừng khi tắt node
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

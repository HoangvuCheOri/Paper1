#!/usr/bin/env python3
"""
Debug node: In liên tục góc Yaw từ cả Encoder và Camera lên terminal.
Dùng để kiểm tra 2 bên có cùng chiều quay không.

Cách dùng:
  1. Bật robot_serial_bridge, state_bridge, camera_node
  2. ros2 run amr_control debug_yaw
  3. Dùng tay xoay robot sang TRÁI (CCW) → cả 2 giá trị phải CÙNG TĂNG
  4. Dùng tay đẩy robot tiến tới → cả 2 giá trị X phải CÙNG TĂNG
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math


class YawDebugger(Node):
    def __init__(self):
        super().__init__('yaw_debugger')

        self.cam_x = 0.0
        self.cam_y = 0.0
        self.cam_yaw = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        self.cam_received = False
        self.odom_received = False

        self.create_subscription(Odometry, '/odom_camera', self.cam_cb, 10)
        self.create_subscription(Odometry, '/odom_raw', self.odom_cb, 10)

        # In kết quả mỗi 0.5 giây
        self.timer = self.create_timer(0.5, self.print_comparison)

        self.get_logger().info(
            "=== YAW DEBUGGER ===\n"
            "  Xoay robot sang TRÁI (CCW): cả 2 Yaw phải CÙNG TĂNG (dương)\n"
            "  Đẩy robot tiến tới: cả 2 X phải CÙNG TĂNG (dương)\n"
            "  Đẩy robot sang trái: cả 2 Y phải CÙNG TĂNG (dương)\n"
            "===================="
        )

    def yaw_from_quat(self, q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def cam_cb(self, msg):
        self.cam_x = msg.pose.pose.position.x
        self.cam_y = msg.pose.pose.position.y
        self.cam_yaw = math.degrees(self.yaw_from_quat(msg.pose.pose.orientation))
        self.cam_received = True

    def odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_yaw = math.degrees(self.yaw_from_quat(msg.pose.pose.orientation))
        self.odom_received = True

    def print_comparison(self):
        if not self.cam_received and not self.odom_received:
            self.get_logger().warn("Chưa nhận được dữ liệu. Kiểm tra các node đã chạy chưa.")
            return

        odom_str = (
            f"X={self.odom_x:+.3f}  Y={self.odom_y:+.3f}  Yaw={self.odom_yaw:+.1f}°"
            if self.odom_received else "--- chưa có ---"
        )
        cam_str = (
            f"X={self.cam_x:+.3f}  Y={self.cam_y:+.3f}  Yaw={self.cam_yaw:+.1f}°"
            if self.cam_received else "--- chưa có ---"
        )

        sign_match = ""
        if self.cam_received and self.odom_received:
            yaw_ok = "✓ CÙNG DẤU" if (self.cam_yaw * self.odom_yaw >= 0) else "✗ NGƯỢC DẤU"
            sign_match = f"  |  Yaw: {yaw_ok}"

        self.get_logger().info(
            f"\n  ODOM: {odom_str}\n"
            f"  CAM:  {cam_str}{sign_match}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = YawDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

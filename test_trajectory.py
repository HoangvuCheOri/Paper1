#!/usr/bin/env python3
"""
Script test: Đo bán kính thực tế từ trajectory overlay
Chạy khi robot đang chạy quỹ đạo tròn
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

class TrajectoryAnalyzer(Node):
    def __init__(self):
        super().__init__('trajectory_analyzer')

        self.odom_sub = self.create_subscription(
            Odometry, '/odom_camera', self.odom_callback, 10
        )

        self.positions = []
        self.start_time = None
        self.last_time = None

        # Threshold: bắt đầu phân tích sau khi có đủ điểm
        self.min_points = 50

        # Timer để phân tích định kỳ
        self.timer = self.create_timer(2.0, self.analyze)

        self.get_logger().info("Trajectory Analyzer started - waiting for data...")

    def odom_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = now
            self.last_time = now

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.positions.append((now, x, y))

        # Chỉ giữ 500 điểm gần nhất
        if len(self.positions) > 500:
            self.positions.pop(0)

    def analyze(self):
        if len(self.positions) < self.min_points:
            self.get_logger().info(f"Need more points: {len(self.positions)}/{self.min_points}")
            return

        # Tính centroid (tâm của quỹ đạo)
        xs = [p[1] for p in self.positions]
        ys = [p[2] for p in self.positions]

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        # Tính khoảng cách trung bình từ tâm
        distances = [math.sqrt((x-cx)**2 + (y-cy)**2) for x, y in zip(xs, ys)]
        avg_radius = sum(distances) / len(distances)

        # Tìm bán kính max và min
        max_radius = max(distances)
        min_radius = min(distances)

        # Ước tính tốc độ dài trung bình
        speeds = []
        for i in range(1, len(self.positions)):
            dt = self.positions[i][0] - self.positions[i-1][0]
            if dt > 0:
                dx = self.positions[i][1] - self.positions[i-1][1]
                dy = self.positions[i][2] - self.positions[i-1][2]
                v = math.sqrt(dx**2 + dy**2) / dt
                speeds.append(v)

        avg_speed = sum(speeds) / len(speeds) if speeds else 0

        # Ước tính chu vi và thời gian một vòng
        if avg_speed > 0:
            circumference = 2 * math.pi * avg_radius
            time_per_lap = circumference / avg_speed if avg_speed > 0 else 0
        else:
            circumference = 0
            time_per_lap = 0

        elapsed = self.positions[-1][0] - self.start_time if self.start_time else 0

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Trajectory Analysis (after {elapsed:.1f}s, {len(self.positions)} pts):")
        self.get_logger().info(f"  Centroid: ({cx:.3f}, {cy:.3f})")
        self.get_logger().info(f"  Avg Radius: {avg_radius:.3f} m")
        self.get_logger().info(f"  Min/Max Radius: {min_radius:.3f} / {max_radius:.3f} m")
        self.get_logger().info(f"  Avg Speed: {avg_speed:.3f} m/s")
        self.get_logger().info(f"  Est. Time per lap: {time_per_lap:.1f} s")
        self.get_logger().info(f"  Target R=0.5m -> Actual R={avg_radius:.3f}m -> Ratio: {avg_radius/0.5:.2f}x")
        self.get_logger().info("=" * 60)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

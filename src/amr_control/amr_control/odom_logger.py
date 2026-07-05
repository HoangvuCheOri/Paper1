import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
import csv
import os
import message_filters

def yaw_from_quaternion(q):
    t3 = 2.0 * (q.w * q.z + q.x * q.y)
    t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)


def normalize_angle_deg(angle):
    return math.degrees(math.atan2(math.sin(math.radians(angle)), math.cos(math.radians(angle))))

def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

class OdomLogger(Node):
    def __init__(self):
        super().__init__('odom_logger')

        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('camera_topic', '/odom_camera')
        self.declare_parameter('sync_queue_size', 20)
        self.declare_parameter('sync_slop', 0.20)
        self.odom_topic = self.get_parameter('odom_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)

        self.odom_sub = message_filters.Subscriber(self, Odometry, self.odom_topic)
        self.cam_sub = message_filters.Subscriber(self, Odometry, self.camera_topic)

        # Sử dụng ApproximateTimeSynchronizer để bắt cặp gói tin có timestamp gần nhau nhất
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.cam_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop
        )
        self.ts.registerCallback(self.sync_cb)

        # Mở file CSV để ghi
        home_dir = os.path.expanduser('~')
        self.file_path = os.path.join(home_dir, 'Paper1', 'odom_compare.csv')
        self.csv_file = open(self.file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Time',
            'Odom_Stamp',
            'Cam_Stamp',
            'Stamp_Diff',
            'Odom_X',
            'Odom_Y',
            'Odom_Yaw(deg)',
            'Cam_X',
            'Cam_Y',
            'Cam_Yaw(deg)',
            'Diff_X',
            'Diff_Y',
            'Diff_Yaw',
        ])

        self.start_time = None
        self.get_logger().info(
            f"Odom Logger started! {self.odom_topic} vs {self.camera_topic}. "
            f"sync_slop={self.sync_slop:.3f}s, ghi data lien tuc vao {self.file_path}"
        )

    def sync_cb(self, odom_msg, cam_msg):
        odom_stamp = stamp_to_sec(odom_msg.header.stamp)
        cam_stamp = stamp_to_sec(cam_msg.header.stamp)
        if odom_stamp <= 0.0:
            odom_stamp = self.get_clock().now().nanoseconds / 1e9
        if cam_stamp <= 0.0:
            cam_stamp = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = odom_stamp

        t = odom_stamp - self.start_time
        stamp_diff = cam_stamp - odom_stamp

        odom_yaw = math.degrees(yaw_from_quaternion(odom_msg.pose.pose.orientation))
        odom_x = odom_msg.pose.pose.position.x
        odom_y = odom_msg.pose.pose.position.y

        cam_yaw = math.degrees(yaw_from_quaternion(cam_msg.pose.pose.orientation))
        cam_x = cam_msg.pose.pose.position.x
        cam_y = cam_msg.pose.pose.position.y

        diff_x = cam_x - odom_x
        diff_y = cam_y - odom_y
        diff_yaw = normalize_angle_deg(cam_yaw - odom_yaw)

        self.csv_writer.writerow([
            f"{t:.2f}",
            f"{odom_stamp:.6f}",
            f"{cam_stamp:.6f}",
            f"{stamp_diff:.6f}",
            f"{odom_x:.3f}", f"{odom_y:.3f}", f"{odom_yaw:.1f}",
            f"{cam_x:.3f}", f"{cam_y:.3f}", f"{cam_yaw:.1f}",
            f"{diff_x:.3f}", f"{diff_y:.3f}", f"{diff_yaw:.1f}"
        ])

        # Tự động flush để ghi liền vào đĩa, dễ theo dõi
        self.csv_file.flush()

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OdomLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

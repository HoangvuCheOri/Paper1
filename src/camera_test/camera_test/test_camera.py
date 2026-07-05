import math
import os
import queue
import threading
import time

import cv2
from geometry_msgs.msg import Pose2D
import numpy as np
import pupil_apriltags as apriltag
import rclpy
from rclpy.node import Node


def normalize_angle_deg(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def yaw_from_rotation_matrix(rot):
    return math.atan2(rot[1, 0], rot[0, 0])


class CameraTestNode(Node):
    def __init__(self):
        super().__init__('camera_test_node')

        self.pose_pub = self.create_publisher(Pose2D, '/apriltag_pose', 10)

        username = os.getenv('CAMERA_USERNAME', 'admin')
        password = os.getenv('CAMERA_PASSWORD', 'lab208b3')
        camera_ip = os.getenv('CAMERA_IP', '192.168.100.56')
        camera_port = os.getenv('CAMERA_PORT', '554')
        self.ip_url = (
            f'rtsp://{username}:{password}@{camera_ip}:{camera_port}'
            '/cam/realmonitor?channel=1&subtype=0'
        )

        self.base_camera_matrix = np.array(
            [
                [767.6786, 0.0, 637.4356],
                [0.0, 765.5082, 357.2588],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.array(
            [-0.2374, 0.0734, 0.00345, -0.00824, -0.0514],
            dtype=np.float32,
        )
        self.zero_dist_coeffs = np.zeros_like(self.dist_coeffs)

        self.detector = apriltag.Detector(
            families='tag36h11',
            nthreads=3,
            refine_edges=1,
        )

        self.marker_size = 0.150  # 15x15 cm
        self.marker_id = 0
        self.marker_3D = np.array(
            [
                [-self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, self.marker_size / 2, 0],
                [self.marker_size / 2, -self.marker_size / 2, 0],
                [-self.marker_size / 2, -self.marker_size / 2, 0],
            ],
            dtype=np.float32,
        )

        self.q = queue.Queue(maxsize=1)
        self.stream_thread = threading.Thread(
            target=self.camera_stream_thread,
            daemon=True,
        )
        self.stream_thread.start()

        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        self.get_logger().info(
            'Camera Test Node Started. Publishing RAW pose to /apriltag_pose'
        )

    def camera_stream_thread(self):
        cap = None
        while rclpy.ok():
            if cap is None or not cap.isOpened():
                self.get_logger().info(f'Connecting to RTSP: {self.ip_url}')
                cap = cv2.VideoCapture(self.ip_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    self.get_logger().warn(
                        'Cannot connect to camera, retrying in 1s...'
                    )
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                continue

            if not self.q.full():
                self.q.put_nowait(frame)
            else:
                self.q.get_nowait()
                self.q.put_nowait(frame)

        if cap:
            cap.release()

    def process_frame(self):
        try:
            frame = self.q.get_nowait()
        except queue.Empty:
            return

        h_in, w_in = frame.shape[:2]
        # Stream có thể bị resize từ 1280x720, nên scale lại intrinsic matrix.
        scale_x = w_in / 1280.0
        scale_y = h_in / 720.0

        camera_matrix = self.base_camera_matrix.copy()
        camera_matrix[0, 0] *= scale_x
        camera_matrix[1, 1] *= scale_y
        camera_matrix[0, 2] *= scale_x
        camera_matrix[1, 2] *= scale_y

        if abs(scale_x - 1.0) < 1e-3 and abs(scale_y - 1.0) < 1e-3:
            frame = cv2.undistort(frame, camera_matrix, self.dist_coeffs)
            dist_coeffs = self.zero_dist_coeffs
        else:
            dist_coeffs = self.dist_coeffs

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        for det in detections:
            if det.tag_id != self.marker_id:
                continue

            img_pts = det.corners.astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                self.marker_3D,
                img_pts,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue

            r_orig, _ = cv2.Rodrigues(rvec)
            yaw_cam = yaw_from_rotation_matrix(r_orig)

            # Khử yaw của tag để lấy độ nghiêng camera so với mặt sàn.
            rz_yaw = np.array(
                [
                    [math.cos(yaw_cam), -math.sin(yaw_cam), 0],
                    [math.sin(yaw_cam), math.cos(yaw_cam), 0],
                    [0, 0, 1],
                ]
            )
            r_floor_to_cam = r_orig @ rz_yaw.T
            pos_floor = r_floor_to_cam.T @ tvec

            robot_x = (-pos_floor[1, 0]) / 1.2
            robot_y = -pos_floor[0, 0]
            yaw_deg = normalize_angle_deg(-(math.degrees(yaw_cam) + 90.0))

            pose_msg = Pose2D()
            pose_msg.x = float(robot_x)
            pose_msg.y = float(robot_y)
            pose_msg.theta = float(yaw_deg)  # Log độ cho dễ đọc
            self.pose_pub.publish(pose_msg)

            self.get_logger().info(
                f'RAW -> X: {robot_x:.3f} m, Y: {robot_y:.3f} m, '
                f'Yaw: {yaw_deg:.1f} deg'
            )

            pt1 = (int(img_pts[0][0]), int(img_pts[0][1]))
            pt2 = (int(img_pts[1][0]), int(img_pts[1][1]))
            pt3 = (int(img_pts[2][0]), int(img_pts[2][1]))
            pt4 = (int(img_pts[3][0]), int(img_pts[3][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.line(frame, pt2, pt3, (0, 255, 0), 2)
            cv2.line(frame, pt3, pt4, (0, 255, 0), 2)
            cv2.line(frame, pt4, pt1, (0, 255, 0), 2)

            tag_center_x = int(np.mean([p[0] for p in img_pts]))
            tag_center_y = int(np.mean([p[1] for p in img_pts]))
            cv2.putText(
                frame,
                f'X:{robot_x:.2f} Y:{robot_y:.2f}',
                (tag_center_x, tag_center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f'Yaw:{yaw_deg:.1f}',
                (tag_center_x, tag_center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            break

        self.draw_camera_origin(frame, w_in, h_in)

        cv2.imshow('Camera Test - AprilTag Raw', frame)
        cv2.waitKey(1)

    def draw_camera_origin(self, frame, w_in, h_in):
        cx, cy = w_in // 2, h_in // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 2)
        cv2.putText(
            frame,
            'ORIGIN (0,0)',
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.arrowedLine(
            frame,
            (cx, cy),
            (cx, cy - 100),
            (255, 0, 0),
            3,
            tipLength=0.1,
        )
        cv2.putText(
            frame,
            '+X',
            (cx - 30, cy - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        cv2.arrowedLine(
            frame,
            (cx, cy),
            (cx - 100, cy),
            (255, 0, 255),
            3,
            tipLength=0.1,
        )
        cv2.putText(
            frame,
            '+Y',
            (cx - 90, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

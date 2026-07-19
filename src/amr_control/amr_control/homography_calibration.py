#!/usr/bin/env python3
"""
Floor Homography Calibration Tool

This script helps calibrate the homography matrix for converting
AprilTag pixel coordinates to real-world floor coordinates.

Usage:
    ros2 run amr_control homography_calibration

Then:
    1. Press 'c' to enter calibration mode
    2. Click on 4 known floor points (in order: TL, TR, BR, BL)
    3. Enter the real-world coordinates when prompted
    4. Press 's' to save the homography matrix
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
import os
import pupil_apriltags as apriltag

IMAGE_POINTS = [  # pixel coordinates measured from overhead camera image
    [932, 363],
    [670, 370],
    [913, 113],
    [662, 114],
    [412, 127],
    [404, 376],
    [411, 637],
    [674, 641],
    [935, 627],
    [804, 366],
    [533, 373],
    [799, 235],
    [532, 244],
    [535, 508],
    [809, 505],
    [671, 508],
    [666, 235],
]

WORLD_POINTS = [  # real-world coordinates in meters (BẠN HÃY ĐIỀN VÀO ĐÂY)
    [1.01, 0.0],  # Tọa độ thực cho điểm [932, 363] 1
    [0.0, 0.0],  # Tọa độ thực cho điểm [670, 370] 2
    [1.01, 1.01],  # Tọa độ thực cho điểm [913, 113] 3
    [0.0, 1.01],  # Tọa độ thực cho điểm [662, 114] 4
    [-1.01, 1.01],  # Tọa độ thực cho điểm [412, 127] 5
    [-1.01, 0.0],  # Tọa độ thực cho điểm [404, 376] 6
    [-1.01, -1.01],  # Tọa độ thực cho điểm [411, 637] 7
    [0.0, -1.01],  # Tọa độ thực cho điểm [674, 641] 8
    [1.01, -1.01],  # Tọa độ thực cho điểm [935, 627] 9
    [0.51, 0.0],  # Tọa độ thực cho điểm [804, 366] 10
    [-0.51, 0.0],  # Tọa độ thực cho điểm [533, 373] 11
    [0.51, 0.51],  # Tọa độ thực cho điểm [799, 235] 12
    [-0.51, 0.51],  # Tọa độ thực cho điểm [532, 244] 13
    [-0.51, -0.51],  # Tọa độ thực cho điểm [535, 508] 14
    [0.51, -0.51],  # Tọa độ thực cho điểm [809, 505] 15
    [0.0, -0.51],  # Tọa độ thực cho điểm [671, 508] 16
    [0.0, 0.51],  # Tọa độ thực cho điểm [666, 235] 17
]


class HomographyCalibrator(Node):
    def __init__(self):
        super().__init__('homography_calibrator')

        self.declare_parameter('output_yaml', 'floor_homography.yaml')
        self.declare_parameter('camera_ip', os.getenv('CAMERA_IP', '192.168.100.56'))
        self.declare_parameter('camera_port', os.getenv('CAMERA_PORT', '554'))
        self.declare_parameter('camera_username', os.getenv('CAMERA_USERNAME', 'admin'))
        self.declare_parameter('camera_password', os.getenv('CAMERA_PASSWORD', 'lab208b3'))
        self.declare_parameter('display_resize', False)
        self.declare_parameter('display_width', 1280)
        self.declare_parameter('display_height', 720)
        self.declare_parameter('undistort_points', True)
        self.declare_parameter('ransac_threshold_m', 0.03)

        self.output_yaml = self.get_parameter('output_yaml').value
        self.display_resize = bool(self.get_parameter('display_resize').value)
        self.display_width = int(self.get_parameter('display_width').value)
        self.display_height = int(self.get_parameter('display_height').value)
        self.undistort_points = bool(self.get_parameter('undistort_points').value)
        self.ransac_threshold_m = max(
            0.001, float(self.get_parameter('ransac_threshold_m').value)
        )
        self.calibration_points = [list(pt) for pt in IMAGE_POINTS]
        self.real_world_points = [list(pt) for pt in WORLD_POINTS]
        self.calibration_mode = False
        self.waiting_for_real_coords = False
        self.current_pixel = None
        self.base_image_width = 1280
        self.base_image_height = 720
        self.frame_width = None
        self.frame_height = None
        self.default_points_scaled = False

        # Camera parameters
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self.ip_url = (
            f"rtsp://{self.get_parameter('camera_username').value}:"
            f"{self.get_parameter('camera_password').value}@"
            f"{self.get_parameter('camera_ip').value}:"
            f"{self.get_parameter('camera_port').value}/cam/realmonitor?channel=1&subtype=0"
        )

        # Calibration camera matrix (at 1280x720)
        self.camera_matrix = np.array([
            [767.6786, 0., 637.4356],
            [0., 765.5082, 357.2588],
            [0., 0., 1.]
        ], dtype=np.float32)
        self.dist_coeffs = np.array(
            [-0.2374, 0.0734, 0.00345, -0.00824, -0.0514],
            dtype=np.float32
        )
        self.marker_size = 0.150
        self.marker_3D = np.array([
            [-self.marker_size/2, self.marker_size/2, 0],
            [self.marker_size/2, self.marker_size/2, 0],
            [self.marker_size/2, -self.marker_size/2, 0],
            [-self.marker_size/2, -self.marker_size/2, 0]
        ], dtype=np.float32)
        self.detector = apriltag.Detector(families='tag36h11', nthreads=3)

        self.window_name = "Homography Calibration"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.get_logger().info("Homography Calibration Tool")
        self.get_logger().info("=" * 50)
        self.get_logger().info("Controls:")
        self.get_logger().info("  'c' - Toggle calibration mode")
        self.get_logger().info("  's' - Save homography to YAML")
        self.get_logger().info("  'r' - Reset calibration points")
        self.get_logger().info("  'x' - Clear all points for a fresh calibration")
        self.get_logger().info("  'q' - Quit")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Output file: {self.output_yaml}")
        self.get_logger().info(
            f"Homography pixel domain: "
            f"{'undistorted' if self.undistort_points else 'distorted (legacy)'}, "
            f"RANSAC threshold={self.ransac_threshold_m:.3f} m"
        )

    def scaled_camera_matrix(self, width, height):
        matrix = self.camera_matrix.copy()
        scale_x = float(width) / float(self.base_image_width)
        scale_y = float(height) / float(self.base_image_height)
        matrix[0, 0] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[0, 2] *= scale_x
        matrix[1, 2] *= scale_y
        return matrix

    def configure_frame_geometry(self, width, height):
        """Put preset 1280x720 points into the actual raw-frame resolution."""
        if self.default_points_scaled:
            return
        self.frame_width = int(width)
        self.frame_height = int(height)
        scale_x = float(width) / float(self.base_image_width)
        scale_y = float(height) / float(self.base_image_height)
        self.calibration_points = [
            [float(point[0]) * scale_x, float(point[1]) * scale_y]
            for point in self.calibration_points
        ]
        self.default_points_scaled = True
        self.get_logger().info(
            f"Calibration frame size: {self.frame_width}x{self.frame_height}"
        )

    def points_for_homography(self, raw_points):
        points = np.asarray(raw_points, dtype=np.float32)
        if not self.undistort_points:
            return points
        if self.frame_width is None or self.frame_height is None:
            raise RuntimeError("camera frame geometry is not initialized")
        matrix = self.scaled_camera_matrix(self.frame_width, self.frame_height)
        return cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            matrix,
            self.dist_coeffs,
            P=matrix,
        ).reshape(-1, 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self.calibration_mode:
            self.get_logger().info("Enable calibration mode with 'c' first")
            return

        raw_x = float(x)
        raw_y = float(y)
        if (
            self.display_resize
            and self.frame_width is not None
            and self.frame_height is not None
        ):
            raw_x *= float(self.frame_width) / max(float(self.display_width), 1.0)
            raw_y *= float(self.frame_height) / max(float(self.display_height), 1.0)

        self.calibration_points.append([raw_x, raw_y])
        self.get_logger().info(
            f"Added point {len(self.calibration_points)}: "
            f"raw_pixel=({raw_x:.1f}, {raw_y:.1f})"
        )

        try:
            coords = input(f"Enter real-world (x,y) for this point: ").strip()
            rx, ry = map(float, coords.split(','))
            self.real_world_points.append([rx, ry])
            self.get_logger().info(f"Added real-world point: ({rx}, {ry})")
        except ValueError:
            self.get_logger().error("Invalid format. Point removed.")
            self.calibration_points.pop()

    def compute_and_save_homography(self):
        if len(self.calibration_points) < 4 or len(self.real_world_points) < 4:
            self.get_logger().error("Need at least 4 pixel and 4 real-world points")
            return
        if len(self.calibration_points) != len(self.real_world_points):
            self.get_logger().error(
                "Pixel/world point counts differ: "
                f"{len(self.calibration_points)} != {len(self.real_world_points)}"
            )
            return

        src_raw = np.array(self.calibration_points, dtype=np.float32)
        dst = np.array(self.real_world_points, dtype=np.float32)
        try:
            src = self.points_for_homography(src_raw)
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            return

        H, mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            self.ransac_threshold_m,
        )
        if H is None:
            self.get_logger().error("Homography computation failed")
            return
            
        H_inv = np.linalg.inv(H)

        # Report both the scientifically relevant floor error and inverse
        # reprojection error in the same pixel domain used to fit H.
        dst_projected = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors_m = np.linalg.norm(dst_projected - dst, axis=1)
        rms_error_m = float(np.sqrt(np.mean(np.square(errors_m))))

        dst_h = np.hstack([dst, np.ones((len(dst), 1))])
        src_proj_h = (H_inv @ dst_h.T).T
        src_proj = src_proj_h[:, :2] / src_proj_h[:, 2:]
        
        errors_px = np.linalg.norm(src - src_proj, axis=1)
        mean_error_px = np.mean(errors_px)
        
        self.get_logger().info(
            f"Floor error: mean={100.0 * float(np.mean(errors_m)):.2f} cm, "
            f"RMS={100.0 * rms_error_m:.2f} cm, "
            f"max={100.0 * float(np.max(errors_m)):.2f} cm"
        )
        self.get_logger().info(f"Mean inverse reprojection error: {mean_error_px:.2f} px")
        if mean_error_px > 5.0:
            self.get_logger().warn("Warning: Mean reprojection error is > 5px. Calibration might be inaccurate.")

        data = {
            'pixel_domain': 'undistorted' if self.undistort_points else 'distorted',
            'image_width': int(self.frame_width),
            'image_height': int(self.frame_height),
            'ransac_threshold_m': float(self.ransac_threshold_m),
            'mean_floor_error_m': float(np.mean(errors_m)),
            'rms_floor_error_m': rms_error_m,
            'max_floor_error_m': float(np.max(errors_m)),
            'camera_model': 'opencv_radtan',
            'camera_matrix': self.scaled_camera_matrix(
                self.frame_width, self.frame_height
            ).tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'src_points': src_raw.tolist(),
            'src_points_homography_domain': src.tolist(),
            'dst_points': self.real_world_points,
            'homography_matrix': H.tolist(),
            'homography_inv_matrix': H_inv.tolist(),
        }

        with open(self.output_yaml, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        self.get_logger().info(f"Saved homography to {self.output_yaml}")
        self.print_homography_info(H, src, dst)

    def print_homography_info(self, H, src, dst):
        self.get_logger().info("=" * 50)
        self.get_logger().info("Homography Matrix H (pixel -> real):")
        for row in H:
            self.get_logger().info(f"  [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")

        # Test a few points
        self.get_logger().info("\nVerification:")
        for i, (px, py) in enumerate(src):
            real = dst[i]
            pixel_h = np.array([[px], [py], [1.0]])
            computed = H @ pixel_h
            computed_xy = (computed[0,0]/computed[2,0], computed[1,0]/computed[2,0])
            self.get_logger().info(
                f"  Pixel({px:.1f}, {py:.1f}) -> Real({computed_xy[0]:.4f}, {computed_xy[1]:.4f}) "
                f"(expected: {real[0]:.4f}, {real[1]:.4f})"
            )

    def run(self):
        import threading
        import queue

        q = queue.Queue(maxsize=1)

        def camera_thread():
            cap = cv2.VideoCapture(self.ip_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while rclpy.ok():
                ret, frame = cap.read()
                if ret and not q.full():
                    q.put_nowait(frame)
                elif not ret:
                    import time
                    time.sleep(0.1)
            cap.release()

        thread = threading.Thread(target=camera_thread, daemon=True)
        thread.start()

        while rclpy.ok():
            try:
                frame = q.get(timeout=1.0)
            except queue.Empty:
                self.get_logger().warn("Camera timeout, retrying...")
                continue

            h_in, w_in = frame.shape[:2]
            self.configure_frame_geometry(w_in, h_in)
            scale_x = w_in / 1280.0
            scale_y = h_in / 720.0

            cam_matrix = self.camera_matrix.copy()
            cam_matrix[0, 0] *= scale_x
            cam_matrix[1, 1] *= scale_y
            cam_matrix[0, 2] *= scale_x
            cam_matrix[1, 2] *= scale_y

            display = frame.copy()

            # Detect AprilTag
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detect(gray)

            for det in detections:
                # Draw detected tag
                pts = det.corners.astype(int)
                for i in range(4):
                    cv2.line(display, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 2)

                # Get tag center
                center = np.mean(det.corners, axis=0)
                cv2.circle(display, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                cv2.putText(display,
                           f"ID:{det.tag_id} ({center[0]:.0f},{center[1]:.0f})",
                           (int(center[0])+10, int(center[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break

            # Draw calibration points
            if self.calibration_mode:
                colors = [(255, 0, 0), (0, 255, 255), (0, 128, 255), (255, 0, 255)]
                for i, pt in enumerate(self.calibration_points):
                    color = colors[i % len(colors)]
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 8, color, -1)
                    cv2.putText(display, str(i+1),
                               (int(pt[0])+10, int(pt[1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if len(self.calibration_points) >= 4:
                    status = f"Ready to save (press 's') - {len(self.calibration_points)} pts"
                    color = (0, 255, 0)
                else:
                    status = f"Click {4-len(self.calibration_points)} more points"
                    color = (0, 255, 255)
                cv2.putText(display, status, (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Mode indicator
            mode = "CALIBRATION" if self.calibration_mode else "VIEW"
            mode_color = (0, 255, 255) if self.calibration_mode else (255, 255, 255)
            cv2.rectangle(display, (5, 5), (200, 35), (0, 0, 0), -1)
            cv2.putText(display, f"Mode: {mode}", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)

            # Instructions
            cv2.putText(display, "c:Calibrate  s:Save  r:Reset  x:Clear  q:Quit",
                       (10, display.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.display_resize:
                display = cv2.resize(
                    display,
                    (self.display_width, self.display_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibration_mode = not self.calibration_mode
                if self.calibration_mode:
                    self.get_logger().info("Calibration mode ON - click 4 floor corners")
                else:
                    self.get_logger().info("Calibration mode OFF")
            elif key == ord('s'):
                self.compute_and_save_homography()
            elif key == ord('r'):
                scale_x = float(self.frame_width) / float(self.base_image_width)
                scale_y = float(self.frame_height) / float(self.base_image_height)
                self.calibration_points = [
                    [float(pt[0]) * scale_x, float(pt[1]) * scale_y]
                    for pt in IMAGE_POINTS
                ]
                self.real_world_points = [list(pt) for pt in WORLD_POINTS]
                self.get_logger().info("Reset calibration points to defaults")
            elif key == ord('x'):
                self.calibration_points = []
                self.real_world_points = []
                self.get_logger().info(
                    "Cleared all calibration points; enable calibration mode "
                    "and add at least four measured correspondences."
                )

        cv2.destroyAllWindows()


def main():
    rclpy.init(args=None)
    calibrator = HomographyCalibrator()
    try:
        calibrator.run()
    except KeyboardInterrupt:
        pass
    finally:
        calibrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

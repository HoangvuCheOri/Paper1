# Hướng dẫn Kiểm tra & Tune EKF Fusion Pipeline

## Tổng quan hệ thống
```
Encoder+IMU (/odom_raw)  ─┐
                           ├──→ [EKF custom_ekf_node] ──→ /odometry/filtered ──→ [BSMC Controller]
Camera AprilTag (/odom_camera) ─┘
```

**State vector:** `[x, y, theta, v, w]`
- Encoder: cung cấp `v` (tốc độ thẳng), `w` (tốc độ góc) cho **prediction**
- Camera: cung cấp `x, y, yaw` tuyệt đối cho **correction**
- IMU: cung cấp `w` (tốc độ góc) cho **correction**

---

## Quy trình test theo thứ tự (KHÔNG nhảy cóc)

### Test 1: Đo nhiễu Camera tĩnh (BẮT BUỘC LÀM TRƯỚC)

**Mục đích:** Đo std_x, std_y, std_yaw khi robot ĐỨNG YÊN → tính R cho EKF.

**Terminal 1:** Bật camera node
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run amr_control camera_node
```

Mặc định `camera_node` publish pose AprilTag đã calib nhưng **không lọc Kalman nội bộ**
(`prefilter_pose=false`). EKF sẽ là nơi lọc/fusion chính.

**Terminal 2:** Chạy script đo variance (mặc định 30 giây)
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run camera_test static_variance_test
```

**Robot phải ĐỨNG YÊN!** Script sẽ thu 30s dữ liệu rồi in ra:
```
╔══════════════════════════════════════════════════════════════╗
║           HỆ SỐ R CHO EKF (variance = std²)                ║
║  camera_x_variance:   0.000XXX                              ║
║  camera_y_variance:   0.000XXX                              ║
║  camera_yaw_variance: 0.000XXX                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Hành động:** Copy 3 số đó vào file `config/custom_ekf.yaml` thay cho giá trị mặc định.

---

### Test 2: Dead-reckoning không camera

**Mục đích:** Xác nhận encoder/IMU ổn trước khi fusion.

```bash
# Tắt hết, chỉ chạy encoder
ros2 run amr_control state_bridge
ros2 run amr_control bsmc_circle
```
Robot chạy tròn 1 vòng R=1m → khi quay về vị trí đầu, đo drift bằng thước:
- Drift < 10 cm/vòng → ✅ OK
- Drift > 20 cm/vòng → ❌ Kiểm tra wheelbase L, encoder PPR

---

### Test 3: EKF Fusion đầy đủ

**Terminal 1:** Camera + State bridge + EKF custom đã load YAML
```bash
ros2 launch amr_control fusion.launch.py
```

Nếu chạy riêng từng node thì **bắt buộc** load params cho EKF:
```bash
ros2 run amr_control custom_ekf_node --ros-args --params-file \
  $(ros2 pkg prefix amr_control)/share/amr_control/config/custom_ekf.yaml
```

**Terminal 3:** Dashboard
```bash
python3 ~/ros2_ws/src/amr_control/dashboard.py
```

**Terminal 4:** Controller
```bash
ros2 run amr_control bsmc_circle
```

**Quan sát Dashboard:**
- `Filtered actual` (EKF) phải mượt hơn `Wheel odom`
- `Filtered actual` phải bám sát `Camera raw` về hình dạng (tròn)
- `Filtered actual` KHÔNG được nhảy theo từng frame camera

---

## Dấu hiệu cần tune

| Triệu chứng | Nguyên nhân | Hành động |
|---|---|---|
| EKF nhảy theo camera mỗi frame | R_camera quá nhỏ (quá tin camera) | Tăng `camera_x/y_variance` gấp đôi |
| EKF vẫn drift sau 3 vòng | R_camera quá lớn (không tin camera) | Giảm `camera_x/y_variance` một nửa |
| EKF không tin camera gì cả | Q quá nhỏ | Tăng `process_noise_x/y` lên |
| EKF không tin encoder | Q quá lớn | Giảm `process_noise_x/y` |
| Camera bị glitch → EKF nhảy | Outlier rejection chưa đủ mạnh | Giảm `max_jump_xy` từ 0.30 xuống 0.20 |
| Camera raw đúng nhưng EKF xoay/lệch mạnh | Camera yaw/frame không cùng chiều với odom hoặc YAML chưa được load | Kiểm tra log `Camera aligned to odom frame`, chạy bằng `fusion.launch.py` |

---

## Cấu trúc file config (custom_ekf.yaml)

```yaml
# Q = Process noise (mô hình encoder sai bao nhiêu mỗi bước)
process_noise_x: 0.0004       # Nhỏ → tin encoder | Lớn → tin camera
process_noise_y: 0.0004
process_noise_theta: 0.0009

# R = Measurement noise (camera đo sai bao nhiêu)
use_camera_msg_covariance: false
camera_x_variance: 0.0225     # ← Thay bằng kết quả Test 1
camera_y_variance: 0.0225
camera_yaw_variance: 0.03

# Outlier rejection
max_jump_xy: 0.30             # Bỏ frame nhảy > 30cm
max_jump_yaw: 0.52            # Bỏ frame nhảy > 30 deg
```

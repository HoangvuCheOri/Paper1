# Chạy thực nghiệm paper không dùng launch

Thiết kế này giữ bốn node cảm biến riêng và chỉ chạy **một** node thực nghiệm
tại một thời điểm. Logger và bộ xuất hình nằm ngay trong node controller hoặc
node kiểm tra ESP-NOW, nên không chạy `paper_data_logger` hay
`paper_dashboard` cùng lúc.

## 1. Build sau khi cập nhật mã nguồn

```bash
cd /home/hoang/Paper1
source /opt/ros/humble/setup.bash
colcon build --packages-select amr_control --symlink-install
source install/setup.bash
```

Mỗi terminal mới đều cần chạy hai lệnh `source` trên.

## 2. Bốn node nền cho thực nghiệm trajectory

Terminal 1 — serial:

```bash
ros2 run amr_control robot_serial_bridge
```

Terminal 2 — state:

```bash
ros2 run amr_control state_bridge
```

Terminal 3 — EKF:

```bash
ros2 run amr_control custom_ekf_node
```

Terminal 4 — camera:

Circle hoặc Square (homography cũ, distorted domain):

```bash
ros2 run amr_control camera_circle_square
```

Figure-eight (homography mới, undistorted domain):

```bash
ros2 run amr_control camera_eight
```

Chỉ chạy một trong hai camera executable tại một thời điểm. `camera_node`
vẫn là entry point tổng quát khi cần truyền đường dẫn YAML thủ công.

Kiểm tra dữ liệu trước khi cho robot chạy:

```bash
ros2 topic hz /odom_camera
ros2 topic hz /odometry/filtered
ros2 topic hz /espnow_link
```

## 3. Chọn đúng một controller

BSMC:

```bash
ros2 run amr_control bsmc_circle
ros2 run amr_control bsmc_eight
ros2 run amr_control bsmc_square
```

Backstepping baseline:

```bash
ros2 run amr_control backstepping_circle
ros2 run amr_control backstepping_eight
ros2 run amr_control backstepping_square
```

Không chạy hai controller cùng lúc vì cả hai đều publish `/cmd_vel`.

Chạy hữu hạn nhiều vòng trong một lệnh bằng `paper_laps`. Ví dụ ba vòng:

```bash
ros2 run amr_control bsmc_eight --ros-args -p paper_laps:=3.0 \
  -p paper_run_id:=eight_bsmc_3laps
```

Circle và Eight dừng theo pha khép kín; Square dừng theo tiến độ quãng đường
thực chiếu lên quỹ đạo, không theo ước lượng thời gian tại các góc.

Với figure-eight, profile mặc định bắt đầu ngay tại điểm giao `(0, 0)` và xoay
đường Lissajous `-45` độ trong hệ camera. Tiếp tuyến tại điểm giao vì vậy trùng
với yaw `0` độ: controller tiến theo `+X` ngay, không quay căn chỉnh tại chỗ.
Hình học và kích thước đường không đổi; hướng `-45` độ này phải được ghi trong
Methods/caption và trường `path_rotation_deg` được lưu trong provenance.

Mặc định node tự dừng sau startup, settling và một vòng quỹ đạo. Ví dụ đặt
run ID, thư mục kết quả và thời lượng cụ thể:

```bash
ros2 run amr_control bsmc_circle --ros-args \
  -p paper_run_id:=circle_bsmc_r01 \
  -p paper_duration:=66.0 \
  -p paper_output_dir:=/home/hoang/Paper1/paper_runs
```

Đặt `paper_duration:=0.0` để chạy đến khi nhấn Ctrl-C (`0` cũng được chấp
nhận bởi phiên bản hiện tại):

```bash
ros2 run amr_control backstepping_circle --ros-args \
  -p paper_run_id:=circle_bs_r01 \
  -p paper_duration:=0.0
```

Chọn square 1 m thay vì profile mặc định 2 m:

```bash
ros2 run amr_control bsmc_square --ros-args \
  -p square_profile:=1m \
  -p paper_run_id:=square_1m_bsmc_r01
```

Khi node kết thúc, nó tự xuất:

- CSV gốc của run;
- `*_trajectory.pdf/png`;
- `*_errors.pdf/png` tính từ AprilTag;
- `*_commands.pdf/png`;
- `*_summary.json` gồm metric và controller parameters thực dùng.

PDF là bản ưu tiên để chèn vào LaTeX; PNG được xuất ở 600 dpi. Không cần chụp
màn hình dashboard.

Sau khi cùng một trajectory có ít nhất một run hợp lệ của cả BSMC và
Backstepping, thư mục `paper_runs/publication` được cập nhật tự động với:

- `*_controller_comparison.pdf/png`: reference, hai controller và AprilTag;
- `*_error_comparison.pdf/png`: bố cục 3×2, cùng y-limit theo từng hàng;
- `*_comparison_provenance.json`: hai run ID/CSV thực sự được chọn;
- `table_tracking_repeats.csv/tex`: tự chuyển sang mean ± standard deviation
  khi có từ hai repeat trở lên.

Các figure này dùng equal aspect, màu controller cố định, không có title trong
hình, PDF vector và PNG 600 dpi. Dữ liệu không bị smoothing.

## 4. Kiểm tra ESP-NOW với số node tối thiểu

ESP-NOW link test chỉ cần hai node:

1. `robot_serial_bridge` để nhận serial và publish `/espnow_link`;
2. `espnow_paper_test` để đo, ghi và xuất hình.

Không cần chạy state, EKF, camera hoặc controller trong link-only test.

Ví dụ static 5 m trong 120 s:

```bash
ros2 run amr_control espnow_paper_test --ros-args \
  -p condition:=static \
  -p distance_m:=5.0 \
  -p duration:=120.0 \
  -p run_id:=static_5m_r01
```

Ví dụ moving 10 m:

```bash
ros2 run amr_control espnow_paper_test --ros-args \
  -p condition:=moving \
  -p distance_m:=10.0 \
  -p duration:=120.0 \
  -p run_id:=moving_10m_r01
```

Kết quả mặc định ở `~/Paper1/paper_runs/espnow`:

- CSV packet-level;
- `*_timeseries.pdf/png`;
- `*_distribution.pdf/png`;
- `*_summary.json` với median, P95, P99, jitter và loss.

Mỗi lần link test kết thúc, `espnow/publication` được cập nhật với Fig. 6 tổng
hợp theo condition/distance và `table_espnow.csv/tex`. Các repeat của cùng một
condition được gộp vào đúng một group.

Nếu firmware gửi packet mở rộng có sequence number, loss được tính từ sequence.
Nếu chỉ nhận packet legacy, summary ghi rõ loss chỉ là ước lượng từ timing gap.

## 5. Quy ước comparison

- Backstepping ép `Ks1=Ks2=0` sau khi nạp cùng trajectory profile.
- BSMC giữ sliding gains đã xác thực; circle dùng `Ks1=0.024`, `Ks2=0.050`
  nếu controller cũ vẫn đang có default bằng zero.
- Giữ nguyên mọi tham số khác giữa một cặp BSMC/Backstepping.
- Dùng cùng `paper_duration`, vị trí đầu, pin, ánh sáng và thứ tự run ngẫu nhiên.
- Nên dùng tối thiểu ba, tốt hơn năm run cho mỗi controller/trajectory.

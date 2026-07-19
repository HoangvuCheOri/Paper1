# SOP thí nghiệm Circle: 5 Backstepping × 5 BSMC

Tài liệu này là quy trình chính thức để tạo dữ liệu, figure và bảng Circle cho
paper. Mỗi controller có **5 run khởi động độc lập**; mỗi run chạy **3 vòng
liên tục**. Đơn vị thống kê là run (`n=5`), không phải 15 lap.

## 1. Điều kiện cố định

- Circle: `R=1.0 m`, `Omega=0.108 rad/s`, 3 lap/run.
- Giữ nguyên mọi tham số ngoài `Ks1`, `Ks2`.
- Backstepping: `Ks1=Ks2=0`.
- BSMC: `Ks1=0.024`, `Ks2=0.050`.
- Cùng vị trí đầu trong dung sai `±2 cm`; cùng yaw trong dung sai `±2 deg`.
- Ghi mức pin trước từng run; giữ AprilTag không bị che và toàn bộ Circle nằm
  trong vùng camera.
- Không thay gain sau khi bắt đầu thu bộ 10 run cuối.

## 2. Node nền

Mỗi lệnh chạy trong một terminal riêng:

```bash
cd /home/hoang/Paper1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run amr_control robot_serial_bridge
```

```bash
cd /home/hoang/Paper1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run amr_control state_bridge
```

```bash
cd /home/hoang/Paper1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run amr_control custom_ekf_node
```

```bash
cd /home/hoang/Paper1
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run amr_control camera_circle_square
```

Trước từng run, kiểm tra `/odom_camera` và `/odometry/filtered` vẫn cập nhật.

## 3. Randomized paired blocks

Khóa lịch trước khi xem kết quả. Lịch cân bằng khuyến nghị:

| Block | Chạy trước | Chạy sau |
|---:|---|---|
| 1 | Backstepping | BSMC |
| 2 | BSMC | Backstepping |
| 3 | Backstepping | BSMC |
| 4 | BSMC | Backstepping |
| 5 | BSMC | Backstepping |

Giữa hai run trong một block: dừng controller, đưa robot về đúng pose ban đầu,
kiểm tra tag và chờ trạng thái đứng yên. Không chạy hai controller đồng thời.

### Backstepping

Thay `XX` bằng `01` đến `05`:

```bash
ros2 run amr_control backstepping_circle --ros-args \
  -p angular_speed:=0.108 \
  -p paper_laps:=3 \
  -p paper_output_dir:=/home/hoang/Paper1/paper_runs/final \
  -p paper_run_id:=final_circle_bs_rXX
```

### BSMC

```bash
ros2 run amr_control bsmc_circle --ros-args \
  -p angular_speed:=0.108 \
  -p paper_laps:=3 \
  -p paper_output_dir:=/home/hoang/Paper1/paper_runs/final \
  -p paper_run_id:=final_circle_bsmc_rXX
```

Node phải tự dừng sau đúng ba vòng và log phải chứa `requested_laps: 3`.

## 4. Tiêu chí loại run được khóa trước

Chỉ loại run khi có ít nhất một lỗi kỹ thuật sau:

- không hoàn thành đủ 3 lap;
- mất/stale AprilTag quá giới hạn audit 0.30 s;
- EKF/odometry timeout;
- robot bị chạm, gặp vật cản hoặc ra ngoài vùng camera;
- thay đổi gain/quỹ đạo ngoài protocol;
- CSV hỏng, thiếu trường hoặc controller label sai.

Không loại run chỉ vì RMSE cao. Ghi lý do loại và chạy lại đúng controller/block.

## 5. Đăng ký 10 CSV

Mở `paper_tools/datasets.yaml`, tìm `nominal_circle`. Dán đường dẫn theo đúng
thứ tự Block 1...5; hai phần tử cùng chỉ số phải thuộc cùng block:

```yaml
nominal_circle:
  required_runs_per_controller: 5
  laps_per_run: 3
  transient_s: 5.0
  camera_max_age_s: 0.30
  message_freshness_s: 0.30
  convergence_threshold_m: 0.05
  convergence_hold_s: 1.0
  backstepping:
    - paper_runs/final/TIMESTAMP_circle_Backstepping_final_circle_bs_r01.csv
    - paper_runs/final/TIMESTAMP_circle_Backstepping_final_circle_bs_r02.csv
    - paper_runs/final/TIMESTAMP_circle_Backstepping_final_circle_bs_r03.csv
    - paper_runs/final/TIMESTAMP_circle_Backstepping_final_circle_bs_r04.csv
    - paper_runs/final/TIMESTAMP_circle_Backstepping_final_circle_bs_r05.csv
  bsmc:
    - paper_runs/final/TIMESTAMP_circle_BSMC_final_circle_bsmc_r01.csv
    - paper_runs/final/TIMESTAMP_circle_BSMC_final_circle_bsmc_r02.csv
    - paper_runs/final/TIMESTAMP_circle_BSMC_final_circle_bsmc_r03.csv
    - paper_runs/final/TIMESTAMP_circle_BSMC_final_circle_bsmc_r04.csv
    - paper_runs/final/TIMESTAMP_circle_BSMC_final_circle_bsmc_r05.csv
```

Không cần ghi summary: pipeline tự tìm file `_summary.json` cùng stem.

## 6. Audit và render

```bash
cd /home/hoang/Paper1
MPLCONFIGDIR=/tmp/bsmc-mpl python3 paper_tools/paper_audit.py
MPLCONFIGDIR=/tmp/bsmc-mpl python3 paper_tools/build_paper_assets.py \
  --output-dir paper_exports
```

Nếu audit có `ERROR`, pipeline dừng và không tạo asset paper mới.

## 7. Asset đầu ra

- `paper_exports/figures/fig2_circle_xy.pdf/png`: vòng hoàn chỉnh cuối của một
  paired block đại diện; block được chọn bằng quy tắc gần median của cả hai
  controller, không chọn run tốt nhất.
- `paper_exports/figures/fig4_circle_errors.pdf/png`: toàn bộ ba lap của paired
  block đại diện; đường chấm đứng đánh dấu ranh giới lap.
- `paper_exports/tables/table1_parameters.tex`: tham số thực đọc từ summary.
- `paper_exports/tables/table4_tracking_performance.tex/csv`: mean ± sample SD
  qua 5 run độc lập.
- `paper_exports/tables/circle_per_lap_metrics.csv`: kiểm tra drift Lap 1–3;
  không dùng mỗi lap như một mẫu độc lập.
- `paper_exports/circle_sop_provenance.json`: danh sách 10 run, block đại diện,
  scope figure và các phép xử lý.
- `paper_exports/data_audit.md`: audit machine-readable cho hồ sơ paper.

PDF vector là bản ưu tiên đưa vào LaTeX; PNG 600 dpi dùng để kiểm tra nhanh.
Pipeline không smoothing và không sửa CSV gốc.

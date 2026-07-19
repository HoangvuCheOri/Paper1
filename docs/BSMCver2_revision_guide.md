# BSMCver2 — Master Plan hoàn thiện paper

Nguồn được đánh giá: **paper_results/BSMCver2.pdf**, 6 trang, ngày 19/07/2026.

Đây là tài liệu duy nhất dùng để:

1. khóa thiết kế thực nghiệm;
2. chạy BSMC–Backstepping comparison;
3. chạy controlled perturbation test;
4. audit và phân tích dữ liệu;
5. tạo figure/table;
6. sửa từng section của paper;
7. kiểm tra trước submission.

## 1. Quyết định khoa học cuối

Paper cần hai nhóm thí nghiệm độc lập:

### Nhóm A — Nominal trajectory tracking

So sánh Backstepping và BSMC trong điều kiện bình thường trên:

- circle;
- figure-eight;
- square 1 m.

Mục tiêu: đánh giá tracking accuracy, variability và control effort.

### Nhóm B — Controlled perturbation

So sánh phản ứng sau một tác động lateral có kiểm soát trên circle.

Mục tiêu: hỗ trợ claim về external-perturbation response. Circle được chọn vì
không có self-intersection hoặc corner switching, nên ít confound hơn
figure-eight và square.

### Claim được phép

Nếu chỉ có Nhóm A:

> The compensated controller yielded lower tracking error under the tested
> nominal hardware and communication conditions.

Nếu Nhóm B cũng cho kết quả tốt:

> Under the tested repeatable lateral perturbation, the compensated controller
> exhibited lower peak tracking error and faster recovery than the nominal
> Backstepping baseline.

Không được kết luận:

> The controller is robust to all external disturbances.

Một robot, một cơ cấu tác động và một môi trường chỉ hỗ trợ claim trong đúng
tested condition.

---

## 2. Các lỗi bắt buộc sửa trong BSMCver2

- [ ] Table I dùng gains cũ, không khớp executable.
- [ ] Circle parameters không khớp run hiện tại.
- [ ] Figure-eight parameters và phương trình không khớp code.
- [ ] Table IV vẫn trống.
- [ ] Fig. 2–4 vẫn là placeholder.
- [ ] Chưa có Fig. 5 square.
- [ ] Convergence metric còn hai giá trị TBD.
- [ ] Table V gọi inter-arrival time là latency.
- [ ] Results chưa phân tích kết quả.
- [ ] Chưa có Discussion đủ mạnh.
- [ ] Abstract và Conclusion đang claim trước khi có bảng.
- [ ] Chưa mô tả đầy đủ trajectory-specific corrections.
- [ ] Paper ghi camera pose 30 Hz nhưng camera timer hiện là 20 Hz.
- [ ] Packet timestamp/sequence description chưa được firmware source chứng minh.
- [ ] References [19], [22], [23] còn YOUR FULL LINK HERE.
- [ ] Cần kiểm tra lại author names, emails và affiliations.

---

## 3. Controller comparison phải là ablation công bằng

Backstepping và BSMC phải dùng chung:

- trajectory generator;
- initial pose;
- k1, k2, k3;
- actuator limits;
- camera và EKF;
- trajectory ramp;
- yaw-bias correction;
- radius feedback;
- figure-eight feedforward shaping;
- square corner logic;
- duration;
- sampling rate.

Khác biệt duy nhất cần chủ động tạo:

~~~text
Backstepping: Ks1 = 0, Ks2 = 0
BSMC:         Ks1 > 0, Ks2 > 0
~~~

Không dùng cặp circle cũ làm comparison cuối vì:

~~~text
old baseline radius_position_gain = 0.4
old BSMC radius_position_gain = 0.5
~~~

Executable mới đã giữ các thành phần khác giống nhau và chỉ tắt sliding terms
trong Backstepping.

### Run naming

~~~text
circle_bs_r01
circle_bsmc_r01
eight_bs_r01
eight_bsmc_r01
square_1m_bs_r01
square_1m_bsmc_r01
circle_push_bs_r01
circle_push_bsmc_r01
~~~

### Randomization

Không chạy toàn bộ Backstepping trước rồi mới chạy BSMC. Dùng paired randomized
blocks:

~~~text
Block 1: BSMC, Backstepping
Block 2: Backstepping, BSMC
Block 3: BSMC, Backstepping
Block 4: Backstepping, BSMC
Block 5: randomized
~~~

Giữa hai run:

1. đưa robot về cùng pose đầu;
2. kiểm tra tag visibility;
3. kiểm tra camera/EKF rate;
4. kiểm tra ESP-NOW;
5. ghi điện áp pin nếu có thể;
6. xác nhận controller và run ID;
7. không thay gains.

---

## 4. Run matrix

### Kế hoạch khuyến nghị

| Scenario | Backstepping | BSMC | Tổng |
|---|---:|---:|---:|
| Circle nominal | 5 | 5 | 10 |
| Figure-eight nominal | 5 | 5 | 10 |
| Square 1 m nominal | 5 | 5 | 10 |
| Circle controlled perturbation | 5 | 5 | 10 |
| Tổng | 20 | 20 | 40 |

Pilot disturbance runs không tính vào 40 validation runs.

### Kế hoạch tối thiểu nếu thiếu thời gian

| Scenario | Backstepping | BSMC | Tổng |
|---|---:|---:|---:|
| Circle nominal | 3 | 3 | 6 |
| Figure-eight nominal | 3 | 3 | 6 |
| Square 1 m nominal | 3 | 3 | 6 |
| Circle controlled perturbation | 3 | 3 | 6 |
| Tổng | 12 | 12 | 24 |

Năm repeat vẫn được khuyến nghị vì ba repeat chỉ cho ước lượng variability yếu.

---

## 5. Controlled perturbation protocol

### 5.1. Điểm quan trọng nhất

Không được vừa để controller hoạt động, vừa đẩy bằng tay, rồi chỉ giữ các run có
camera displacement bằng nhau.

Lý do: displacement sau push đã chứa phản ứng của controller. Một controller
phục hồi nhanh có thể phải nhận lực lớn hơn mới đạt cùng 15 cm. Khi đó hai
controller không nhận cùng disturbance input và kết quả bị selection bias.

Phải chọn đúng một trong hai protocol sau.

### 5.2. Protocol A — Active force-matched perturbation

Đây là protocol ưu tiên nếu paper muốn dùng từ disturbance rejection.

Controller hoạt động liên tục. Hai controller nhận disturbance input tương
đương:

- cùng peak force;
- cùng force impulse;
- cùng duration;
- cùng hướng;
- cùng desired phase;
- cùng điểm tiếp xúc trên robot.

Phương án thiết bị:

1. spring-loaded lateral pusher;
2. thanh đẩy có force gauge/load cell;
3. con lắc có khối lượng, chiều dài và release angle cố định.

Thông số bắt đầu cho pilot:

~~~text
trajectory = circle
radius = 1.0 m
angular speed = 0.108 rad/s
direction = counter-clockwise
push direction = lateral outward
target force = 5.0 N
force tolerance = plus/minus 0.5 N
target duration = 0.50 s
duration tolerance = plus/minus 0.10 s
target desired phase = pi
phase tolerance = plus/minus 0.10 rad
camera freshness <= 0.30 s
~~~

Nếu có force samples:

\[
J=\int_{t_{\rm on}}^{t_{\rm off}}F(t)\,dt.
\]

Force impulse J là đại lượng tốt hơn peak force đơn lẻ để chứng minh input giữa
hai controller tương đương.

Camera displacement, peak tracking error và recovery time là outcomes. Không
dùng chúng để điều chỉnh lực riêng theo controller.

### 5.3. Protocol B — Paused displacement-matched recovery

Dùng khi không có force instrumentation đáng tin cậy.

Quy trình:

1. chạy đến desired phase cố định;
2. pause controller và gửi zero velocity;
3. reposition robot lateral outward;
4. dùng AprilTag xác nhận displacement 0.15 plus/minus 0.03 m;
5. thả robot hoàn toàn;
6. đánh dấu resume event;
7. resume controller;
8. đo recovery.

Protocol này so sánh recovery từ cùng initial pose error. Nó không phải bằng
chứng hai controller chống lại cùng external force.

Cách gọi đúng:

> recovery from a controlled lateral pose perturbation

Không gọi:

> rejection of an identical external force

### 5.4. Chọn protocol

- Có force gauge/load cell/pusher/con lắc: chọn Protocol A.
- Chỉ có camera và thao tác tay: chọn Protocol B.
- Không trộn Protocol A và B trong cùng mean/table.
- Không dùng active hand push và lọc run theo displacement.

### 5.5. Pilot phase

Chạy ba pilot runs không đưa vào kết quả:

1. xác nhận robot không ra ngoài field of view;
2. xác nhận tag không bị che;
3. xác nhận tác động lớn hơn camera noise;
4. xác nhận robot vẫn có khả năng recovery;
5. chọn force/displacement cuối;
6. khóa mọi threshold trước validation.

Không thay force hoặc recovery threshold sau khi đã xem validation results.

---

## 6. Push timing và event marking

### 6.1. Không dùng process time

Không dùng T push bằng 15 s tính từ lúc process khởi tạo. Startup, EKF settling
và trajectory ramp có thể làm event lệch giữa run.

Trigger theo:

- trajectory time; hoặc
- desired phase.

Khuyến nghị circle:

~~~text
target phase = pi
phase tolerance = plus/minus 0.10 rad
minimum steady tracking before event = 5 s
~~~

Với Omega bằng 0.108 rad/s, phase pi xuất hiện khoảng 29.1 s sau motion phase
nếu bỏ qua ramp offset. Event thật vẫn phải lấy từ desired phase, không lấy từ
ước lượng thời gian này.

### 6.2. Event source

Không dùng camera jump lớn hơn 5 cm làm event source chính. Nó có thể:

- bỏ sót push diễn ra qua nhiều frames;
- trigger nhầm vì camera dropout/jitter;
- lệch theo tốc độ phục hồi của controller.

Event source chính:

- force threshold trigger;
- physical switch;
- explicit key/topic marker.

Automatic camera detection chỉ dùng làm consistency check.

### 6.3. Event fields

Protocol A:

~~~text
disturbance_armed
disturbance_onset
disturbance_end
target_force_n
measured_force_n
force_impulse_ns
application_direction
desired_phase_at_onset
~~~

Protocol B:

~~~text
control_paused
target_displacement_reached
robot_released
control_resumed
target_displacement_m
measured_displacement_m
desired_phase_at_resume
~~~

Để giữ số node thấp, event capture nên được tích hợp vào controller/capture
process. Một ROS topic one-shot cũng chấp nhận được nhưng logger phải subscribe
và ghi chính xác timestamp.

---

## 7. Cách tính perturbation displacement

### 7.1. Không dùng raw before–after distance khi robot đang chạy

Trong Protocol A, không dùng:

\[
\sqrt{(x_{\rm after}-x_{\rm before})^2+
(y_{\rm after}-y_{\rm before})^2}.
\]

Robot vẫn tự tiến dọc trajectory, nên raw distance trộn forward motion với
lateral perturbation.

### 7.2. Circle radial deviation

Với tâm circle \((x_c,y_c)\):

\[
e_r(t)=
\sqrt{(x_{\rm cam}(t)-x_c)^2+
(y_{\rm cam}(t)-y_c)^2}-R.
\]

Incremental outward displacement:

\[
\Delta e_r=
\max_{t\in[t_{\rm on},t_{\rm on}+T_p]}e_r(t)
-{\rm median}_{t\in[t_{\rm on}-T_{\rm pre},t_{\rm on})}e_r(t).
\]

Khuyến nghị:

~~~text
pre-event median window = 0.30 s
peak search window = 1.0 s
~~~

### 7.3. General signed cross-track error

\[
e_\perp(t)=
-\sin\theta_d(t)[x_{\rm cam}(t)-x_d(t)]
+\cos\theta_d(t)[y_{\rm cam}(t)-y_d(t)].
\]

Outward sign phải được định nghĩa trước. Không dùng một camera frame đơn lẻ;
dùng median window để giảm jitter.

---

## 8. CSV schema cuối

### Timing và metadata

| Field | Nội dung |
|---|---|
| t | thời gian từ lúc capture bắt đầu |
| trajectory_time | thời gian sau startup/settling |
| ros_time | ROS timestamp |
| controller | Backstepping hoặc BSMC |
| trajectory | circle/eight/square |
| run_id | unique run ID |
| protocol | nominal/active-force/paused-displacement |

### Pose và reference

| Field | Nội dung |
|---|---|
| odom_x, odom_y, odom_yaw | EKF pose dùng cho feedback |
| camera_stamp | timestamp camera |
| camera_age_s | camera message age |
| camera_x, camera_y, camera_yaw | raw AprilTag pose |
| desired_x, desired_y, desired_yaw | reference |

### Errors

| Field | Nội dung |
|---|---|
| error_ex, error_ey, error_etheta | controller/EKF errors |
| camera_error_ex | camera-derived longitudinal error |
| camera_error_ey | camera-derived lateral error |
| camera_error_etheta | camera-derived heading error |
| camera_error_ep | camera-derived position norm |

### Command và measured velocity

| Field | Nội dung |
|---|---|
| cmd_v, cmd_w | velocity commands |
| odom_v, odom_w | measured/fused velocity |

### Perturbation

| Field | Nội dung |
|---|---|
| disturbance_event | armed/onset/end/pause/resume |
| desired_phase | phase tại event |
| target_force_n | Protocol A |
| measured_force_n | Protocol A |
| force_impulse_ns | Protocol A |
| target_displacement_m | Protocol B |
| measured_displacement_m | Protocol B |
| operator_note | protocol deviation |

### Link health

| Field | Nội dung |
|---|---|
| espnow_seq | packet sequence |
| espnow_interarrival_ms | packet inter-arrival |
| espnow_seq_gap | missing packet indicator |

Camera node hiện đặt timer khoảng 20 Hz, không phải 30 Hz. Paper chỉ được báo
rate sau khi đo bằng topic hz.

---

## 9. Disturbance metrics

Primary metrics:

1. peak incremental camera-derived position error;
2. recovery time;
3. post-event integrated absolute error.

Secondary metrics:

1. peak heading error;
2. radial/cross-track displacement;
3. overshoot count;
4. v saturation ratio;
5. omega saturation ratio;
6. number of unrecovered runs.

### 9.1. Pre-event baseline

\[
\bar e_{\rm pre}
=\frac{1}{T_{\rm pre}}\int_{t_d-T_{\rm pre}}^{t_d}e_p(t)\,dt,
\qquad T_{\rm pre}=3\ {\rm s}.
\]

### 9.2. Peak incremental error

\[
e_{\rm peak}
=\max_{t\in[t_d,t_d+10]}e_p(t)-\bar e_{\rm pre}.
\]

### 9.3. IAE

\[
{\rm IAE}_{10}
=\int_{t_d}^{t_d+10}|e_p(t)-\bar e_{\rm pre}|\,dt.
\]

### 9.4. Recovery time

\[
T_{\rm rec}
=\min\{t-t_d:e_p(\tau)\le \bar e_{\rm pre}+\epsilon_r,
\ \forall\tau\in[t,t+T_{\rm hold}]\}.
\]

Pilot values:

~~~text
T_pre = 3 s
post-event window = 10 s
epsilon_r = 0.03 m above pre-event baseline
T_hold = 1.0 s
~~~

Khóa các giá trị sau pilot, trước validation.

Nếu robot không recovery trong 10 s:

- không xóa run;
- ghi recovery time là censored hoặc lớn hơn 10 s;
- báo số unrecovered runs.

### 9.5. Overshoot count

Dùng hysteresis:

~~~text
enter recovered band: e_p <= threshold
exit recovered band: e_p >= threshold + 0.01 m
minimum dwell = 0.20 s
~~~

### 9.6. Saturation

Không kiểm tra equality tuyệt đối:

~~~text
v saturated when abs(cmd_v) >= 0.98 times max_v
w saturated when abs(cmd_w) >= 0.98 times max_w
~~~

---

## 10. Invalidity và protocol deviations

Predeclare invalid run:

- sai controller parameters;
- tag mất quá 0.30 s trong primary recovery window;
- camera age vượt limit liên tục;
- force/impulse ngoài acceptance window của Protocol A;
- initial displacement ngoài tolerance của Protocol B;
- emergency stop;
- collision;
- event timestamp bị thiếu.

Packet gap trùng push không tự động làm run invalid:

1. ghi protocol deviation;
2. chạy sensitivity analysis có/không có run đó;
3. không âm thầm loại sau khi xem kết quả.

Không xóa failed runs. Run sheet phải ghi:

- run ID;
- controller;
- protocol;
- accept/reject;
- reason;
- camera health;
- link health;
- operator note.

---

## 11. Normal tracking metrics

Table tracking nên báo:

| Controller | Trajectory | n | Position RMSE | Path RMSE | Heading RMSE | Max error |
|---|---|---:|---:|---:|---:|---:|

Đơn vị:

- position/path/max error: cm;
- heading: degree.

Với n lớn hơn hoặc bằng 2:

\[
\bar{x}\pm s,
\]

trong đó s là sample standard deviation.

Với n bằng 1, báo giá trị đơn và ghi rõ single-run result.

Primary evaluation source nên là camera-derived errors với freshness limit.
Không gọi AprilTag independent ground truth tuyệt đối vì camera cũng tham gia
EKF feedback.

Khuyến nghị bỏ convergence column khỏi nominal tracking table. Recovery time
đã được định nghĩa rõ hơn trong perturbation experiment.

---

## 12. ESP-NOW protocol và terminology

Giá trị khoảng 50 ms hiện là inter-arrival time, không phải one-way latency.

Sửa:

~~~text
Median lat.  -> Median inter-arrival
Jitter (95%) -> P95 inter-arrival
~~~

Table:

| Condition | Distance | n | Packets | Median | P95 | P99 | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|

- Có sequence: loss từ sequence gaps.
- Không có sequence: ghi gap-estimated loss.
- Không có synchronized clocks: không báo one-way latency.
- Nếu cần latency: đo RTT hoặc mô tả clock synchronization.

Paper hiện nói CMD/DATA có timestamp và sequence nhưng repository chưa chứng
minh đầy đủ. Cần lưu source firmware ESP32 base, rover và STM32 đúng bản đã
flash, cùng exact packet format.

---

## 13. Đồng bộ Methodology với code

### Circle

Paper hiện ghi Omega bằng 0.125 rad/s nhưng executable gần 0.108 rad/s.

Controller còn có:

- trajectory ramp;
- radius-position correction;
- radius feedback;
- yaw-bias correction.

### Figure-eight

Paper hiện dùng A bằng 0.5, B bằng 0.25, Omega bằng 0.10. Executable hiện gần:

\[
x=A\sin(\Omega t),\qquad
y=A\sin(\Omega t)\cos(\Omega t),
\]

với A bằng 1.0 m và Omega bằng 0.07 rad/s.

Controller còn có:

- ramp;
- entry-heading blend;
- direction-dependent angular feedforward;
- yaw-rate feedback;
- center-dependent k1.

### Square

Paper phải mô tả:

- side length;
- desired/corner speed;
- corner deceleration;
- minimum forward speed;
- sharp-corner angular correction;
- damping;
- projection-based progress;
- discrete 90-degree heading switching.

Không gọi desired heading smooth nếu reference vẫn đổi rời rạc.

### Text bắt buộc

> All trajectory-specific shaping, actuator limits, localization settings,
> and shared engineering corrections were held identical between the two
> controllers. The nominal Backstepping ablation differed only by setting
> \(K_{s1}=K_{s2}=0\).

---

## 14. Table I cần tạo lại

Thông số gần với executable:

| Profile | k1 | k2 | k3 | Ks1 | Ks2 |
|---|---:|---:|---:|---:|---:|
| Circle BSMC | 1.163 | 4.499 | 3.300 | 0.024 | 0.050 |
| Figure-eight BSMC | 0.220584 | 6.5 | 7.0 | 0.036833 | 0.115911 |
| Square BSMC | 0.80 | profile-specific | 7.0 | 0.08 | 0.10 |

Limits phổ biến:

~~~text
max_v = 0.18 m/s
max_w = 0.85 rad/s
~~~

Không nhập bảng bằng tay. Lấy parameters từ summary JSON của validation runs.
Thêm square profile và các trajectory-specific parameters cần cho
reproducibility.

---

## 15. Figure plan

### Fig. 1 — Architecture

- xuất vector;
- tăng kích thước chữ;
- loại label không cần thiết;
- đảm bảo đọc được ở final column width.

### Fig. 2 — Circle comparison

- reference dashed black;
- Backstepping orange;
- BSMC blue;
- raw AprilTag low-alpha dots;
- equal aspect;
- start marker;
- caption đúng R và Omega;
- không có in-figure title.

### Fig. 3 — Figure-eight comparison

- cùng color mapping;
- alpha để thấy self-intersection;
- equal aspect;
- caption đúng amplitude và angular speed.

### Fig. 4 — Circle error comparison

- 3 rows × 2 columns;
- left Backstepping, right BSMC;
- ex, ey, e-theta;
- shared y-limit per row;
- camera-derived error;
- không smooth.

### Fig. 5 — Square comparison

- bắt buộc;
- equal aspect;
- mark four vertices;
- show corner overshoot;
- caption ghi side length/profile;
- Discussion giải thích heading spikes.

### Fig. 6 — ESP-NOW

- box/violin distribution;
- static/moving tại 5, 10, 20 m;
- nominal 50 ms line;
- loss annotations;
- repeats cùng condition được gộp.

### Fig. 7 — Perturbation response

Khuyến nghị 3 rows × 2 columns:

- left Backstepping;
- right BSMC;
- row 1 position error;
- row 2 heading error;
- row 3 angular command;
- same y-limit;
- event vertical line;
- mean plus/minus one standard deviation band.

### Fig. 8 — Hardware setup

Optional nếu thiếu trang. Perturbation figure quan trọng hơn setup photo cho
robustness claim.

---

## 16. Result tables

### Normal tracking

| Controller | Trajectory | n | Position RMSE | Path RMSE | Heading RMSE | Max error |
|---|---|---:|---:|---:|---:|---:|

### Protocol A disturbance

| Controller | n | Force impulse | Peak error | Recovery | IAE10 | Unrecovered |
|---|---:|---:|---:|---:|---:|---:|

### Protocol B recovery

| Controller | n | Initial displacement | Peak error | Recovery | IAE10 | Unrecovered |
|---|---:|---:|---:|---:|---:|---:|

### ESP-NOW

| Condition | Distance | n | Packets | Median IA | P95 | P99 | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|

Không gộp Protocol A và B vào cùng mean.

---

## 17. Methodology text mẫu

### Protocol A

> A repeatable lateral perturbation was applied during steady-state circle
> tracking at a fixed desired phase. A calibrated pusher applied an outward
> force of 5.0 plus/minus 0.5 N for 0.50 plus/minus 0.10 s at the same marked
> contact point. The controller remained active throughout the event. Force,
> event timing, AprilTag pose, EKF state, reference pose, commands, and
> ESP-NOW diagnostics were logged. Five independent trials were performed per
> controller in randomized paired blocks.

### Protocol B

> Control was paused at a fixed desired phase and the robot was displaced
> laterally outward by 0.15 plus/minus 0.03 m, measured using the AprilTag pose.
> The robot was released and control was resumed. This experiment evaluates
> recovery from a matched pose error rather than rejection of an identical
> applied force.

---

## 18. Results text mẫu

### Normal tracking

> Across five circle trials, compensated BSMC reduced the camera-derived
> position RMSE from XX.XX plus/minus X.XX cm to YY.YY plus/minus Y.YY cm,
> corresponding to a ZZ.Z percent reduction.

### Protocol A

> The applied disturbance input was comparable between controllers, with force
> impulse values of XX plus/minus XX N s for Backstepping and YY plus/minus YY
> N s for BSMC. BSMC reduced the peak incremental position error from ... to
> ... cm and recovery time from ... to ... s.

### Protocol B

> Initial lateral displacement was comparable between controllers. From this
> matched pose error, BSMC exhibited ... peak error and ... recovery time,
> compared with ... for Backstepping.

Không dùng significantly nếu chưa có statistical test. Có thể dùng:

- yielded a lower mean;
- consistently reduced;
- exhibited faster recovery.

Nếu đủ repeat, cân nhắc paired confidence intervals hoặc Wilcoxon signed-rank.
Không chọn statistical test sau khi thấy kết quả.

---

## 19. Discussion cần có

1. Tại sao BSMC giúp hoặc không giúp từng trajectory.
2. Backstepping có thể ngang hoặc tốt hơn trong nominal condition.
3. BSMC value phải được đánh giá rõ dưới perturbation.
4. Square heading discontinuity và corner spikes.
5. Figure-eight self-intersection behavior.
6. Actuator saturation.
7. Camera noise/occlusion.
8. ESP-NOW gaps.
9. Camera tham gia feedback nên không phải independent ground truth.
10. Lateral push không chứng minh robustness với mọi disturbance.
11. Không có strict sliding reaching/UUB proof.
12. Generalization bị giới hạn bởi một robot và một environment.

---

## 20. Abstract và Conclusion

### Abstract

Chỉ sửa sau khi khóa bảng:

- ghi số run;
- đưa 2–3 con số chính;
- dùng inter-arrival thay latency;
- không dùng highly reliable nếu loss gần 6 phần trăm;
- chỉ claim perturbation response nếu đã chạy test.

### Conclusion

- thêm số liệu định lượng;
- không lặp Abstract;
- nêu scope;
- nêu limitations;
- dùng sliding-mode-inspired;
- không overclaim robustness;
- không gọi inter-arrival là latency.

---

## 21. References và metadata

- [ ] Thay YOUR FULL LINK HERE.
- [ ] Kiểm tra [19], [22], [23].
- [ ] Dùng DOI/publisher page khi có.
- [ ] Dẫn trực tiếp ESP-NOW documentation.
- [ ] Kiểm tra author/title/venue/year/pages.
- [ ] Kiểm tra author names và emails.
- [ ] Kiểm tra affiliations.
- [ ] Chuẩn hóa Vietnam.
- [ ] Kiểm tra corresponding-author mark.

---

## 22. Trình tự thực hiện

### Phase A — Freeze

1. khóa controller profiles;
2. chọn Protocol A hoặc B;
3. chạy pilot;
4. khóa disturbance level;
5. khóa thresholds/analysis windows;
6. tạo randomized run sheet;
7. không tune thêm.

### Phase B — Collect

1. chạy normal tracking matrix;
2. chạy perturbation matrix;
3. chạy ESP-NOW repeats;
4. backup raw logs;
5. lưu firmware source/version;
6. lưu setup photo/video.

### Phase C — Audit

1. kiểm tra camera freshness;
2. kiểm tra event markers;
3. kiểm tra force/displacement acceptance;
4. kiểm tra parameters;
5. ghi protocol deviations;
6. khóa analysis set;
7. không cherry-pick.

### Phase D — Generate

1. tạo vector figures;
2. tạo tables;
3. kiểm tra provenance;
4. kiểm tra mean/std và n;
5. kiểm tra equal aspect/shared scales.

### Phase E — Write

1. sửa Methodology;
2. sửa Table I;
3. viết Normal Results;
4. viết Perturbation Results;
5. viết ESP-NOW Results;
6. viết Discussion;
7. cập nhật Abstract;
8. cập nhật Conclusion;
9. sửa References.

---

## 23. Commands chính

Build:

~~~bash
cd /home/hoang/Paper1
source /opt/ros/humble/setup.bash
colcon build --packages-select amr_control --symlink-install
source install/setup.bash
~~~

Node nền chạy riêng:

~~~bash
ros2 run amr_control robot_serial_bridge
ros2 run amr_control state_bridge
ros2 run amr_control custom_ekf_node
ros2 run amr_control camera_node
~~~

Controller:

~~~bash
ros2 run amr_control backstepping_circle
ros2 run amr_control bsmc_circle
ros2 run amr_control backstepping_eight
ros2 run amr_control bsmc_eight
ros2 run amr_control backstepping_square --ros-args -p square_profile:=1m
ros2 run amr_control bsmc_square --ros-args -p square_profile:=1m
~~~

Rate checks:

~~~bash
ros2 topic hz /odom_camera
ros2 topic hz /odometry/filtered
ros2 topic hz /espnow_link
~~~

---

## 24. Submission gate

Chỉ submission khi:

- [ ] không còn placeholder/TBD/table trống;
- [ ] Table I khớp summary JSON;
- [ ] trajectory definitions khớp executable;
- [ ] comparison chỉ khác Ks terms;
- [ ] đủ repeat;
- [ ] perturbation input hoặc initial displacement được balance;
- [ ] event timestamps đầy đủ;
- [ ] invalidity rules được áp dụng trước khi xem kết quả;
- [ ] có square figure và corner discussion;
- [ ] inter-arrival không bị gọi là latency;
- [ ] Abstract/Conclusion khớp tables;
- [ ] không có anomaly chưa giải thích;
- [ ] không còn placeholder URL;
- [ ] figures đọc được ở final IEEE size;
- [ ] provenance ghi đúng run ID và CSV;
- [ ] claims không vượt quá tested conditions.


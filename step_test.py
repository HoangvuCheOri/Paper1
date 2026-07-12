#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

class StepTestNode(Node):
    def __init__(self):
        super().__init__('step_test_node')
        
        self.declare_parameter('test_type', 'linear') 
        self.declare_parameter('target_vel', 0.15)
        self.declare_parameter('duration', 3.0)
        self.declare_parameter('runs', 5)
        
        self.test_type = self.get_parameter('test_type').value
        self.base_target_vel = float(self.get_parameter('target_vel').value)
        self.duration = float(self.get_parameter('duration').value)
        self.max_runs = int(self.get_parameter('runs').value)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom_raw', self.odom_cb, 10)
        
        self.current_run = 0
        self.taus = []
        
        self.state = 'INIT'
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"Bắt đầu bài test {self.max_runs} lần. Sẽ tự động ĐẢO CHIỀU tiến/lùi để tránh robot đâm tường.")
        self.reset_run()
        
    def reset_run(self):
        self.current_run += 1
        # Đảo chiều sau mỗi lần chạy: Lần 1 dương, lần 2 âm, ...
        sign = 1.0 if self.current_run % 2 != 0 else -1.0
        self.target_vel = self.base_target_vel * sign
        self.target_63_abs = abs(0.632 * self.target_vel)
        
        self.start_time = 0.0
        self.t_63 = 0.0
        self.found_tau = False
        
        self.state = 'INIT'
        self.init_time = time.time()
        self.get_logger().info(f"--- Chuẩn bị chạy lần {self.current_run}/{self.max_runs} (Mục tiêu: {self.target_vel:.3f}) ---")
        
    def control_loop(self):
        now = time.time()
        
        if self.state == 'INIT':
            if now - self.init_time > 1.5:  # Đợi 1.5s cho xe hoàn toàn đứng im trước khi test
                self.state = 'RUNNING'
                self.start_time = now
                self.get_logger().info(f">>> LẦN {self.current_run}: BƠM STEP COMMAND! <<<")
        
        elif self.state == 'RUNNING':
            cmd = Twist()
            if self.test_type == 'linear':
                cmd.linear.x = self.target_vel
            else:
                cmd.angular.z = self.target_vel
            self.cmd_pub.publish(cmd)
            
            if now - self.start_time > self.duration:
                self.state = 'STOPPING'
                self.stop_time = now
                self.get_logger().info(f">>> LẦN {self.current_run}: DỪNG! <<<")
                if not self.found_tau:
                    self.get_logger().warn("Lần này không đạt được 63.2% vận tốc.")
                
        elif self.state == 'STOPPING':
            self.cmd_pub.publish(Twist()) # pub 0 liên tục để hãm phanh
            if now - self.stop_time > 1.5: # Đợi 1.5s cho xe phanh hẳn
                if self.current_run < self.max_runs:
                    self.reset_run()
                else:
                    self.print_results()
                    import sys
                    sys.exit(0)
                
    def odom_cb(self, msg):
        if self.state != 'RUNNING' or self.found_tau:
            return
            
        now = time.time()
        current_vel = msg.twist.twist.linear.x if self.test_type == 'linear' else msg.twist.twist.angular.z
        
        if abs(current_vel) >= self.target_63_abs:
            self.t_63 = now
            self.found_tau = True
            tau = self.t_63 - self.start_time
            self.taus.append(tau)
            bw = 1.0 / tau if tau > 0 else 0
            self.get_logger().info(f"*** LẦN {self.current_run}: ĐẠT 63.2% tại t = {tau:.3f}s (BW = {bw:.2f} rad/s) ***")
            
    def print_results(self):
        print("\n" + "="*60)
        print("🏆 KẾT QUẢ TRUNG BÌNH STEP TEST (" + str(self.max_runs) + " lần):")
        print(f"   - Loại test (Type):      {self.test_type}")
        print(f"   - Vận tốc gốc:           {self.base_target_vel}")
        
        if len(self.taus) > 0:
            avg_tau = sum(self.taus) / len(self.taus)
            avg_bw = 1.0 / avg_tau if avg_tau > 0 else 0
            
            print(f"   - Các mốc tau ghi nhận:  {[round(t, 4) for t in self.taus]}")
            print(f"   - Hằng số thời gian TB:  {avg_tau:.4f} s")
            print(f"   - Băng thông TB:         {avg_bw:.4f} rad/s")
            print("\n👉 Hãy điền giá trị Băng thông TB này vào TABLE II trong paper!")
        else:
            print("   - THẤT BẠI: Cả " + str(self.max_runs) + " lần đều không đạt vận tốc. Kiểm tra lại thông số!")
        print("="*60 + "\n")

def main(args=None):
    rclpy.init(args=args)
    node = StepTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

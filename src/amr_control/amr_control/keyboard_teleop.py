#!/usr/bin/env python3
"""
keyboard_teleop.py — Điều khiển robot bằng bàn phím (WASD / Mũi tên)

Không cần cài thêm thư viện — dùng termios/tty (có sẵn trong Linux).

Cơ chế "đè phím = đi, bỏ ra = dừng":
  - Terminal đọc ký tự raw với timeout 120ms
  - Khi giữ phím: OS tự gửi ký tự liên tục → robot tiếp tục đi
  - Khi bỏ tay: timeout hết → gửi lệnh dừng

Phím điều khiển:
  W / ↑       : Tiến thẳng
  S / ↓       : Lùi
  A / ←       : Quay trái
  D / →       : Quay phải
  W+A / W+D   : Tiến + rẽ (nhấn luân phiên nhanh)
  Q / ESC     : Thoát
"""

import sys
import tty
import termios
import select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


# Tốc độ tối đa
MAX_V      = 0.20   # m/s
MAX_W      = 0.60   # rad/s
TIMEOUT_S  = 0.12   # Giây không có phím → dừng robot


def get_key(timeout=TIMEOUT_S):
    """Đọc 1 ký tự từ stdin với timeout. Trả về '' nếu hết timeout."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            ch = sys.stdin.read(1)
            # Xử lý escape sequence (mũi tên: ESC [ A/B/C/D)
            if ch == '\x1b':
                ready2, _, _ = select.select([sys.stdin], [], [], 0.02)
                if ready2:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ready3, _, _ = select.select([sys.stdin], [], [], 0.02)
                        if ready3:
                            ch3 = sys.stdin.read(1)
                            arrow_map = {'A': 'w', 'B': 's', 'C': 'd', 'D': 'a'}
                            return arrow_map.get(ch3, '')
                return 'q'   # ESC đơn → thoát
            return ch
        return ''   # Timeout
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Ánh xạ phím → (v, w)
KEY_MAP = {
    'w': ( MAX_V,    0.0),
    's': (-MAX_V,    0.0),
    'a': ( 0.0,      MAX_W),
    'd': ( 0.0,     -MAX_W),
}

HELP = """
╔══════════════════════════════════════════════╗
║       KEYBOARD TELEOP — AMR Robot           ║
╠══════════════════════════════════════════════╣
║  W / ↑  : Tiến    S / ↓  : Lùi             ║
║  A / ←  : Trái    D / →  : Phải            ║
║  Q / ESC: Thoát                             ║
╠══════════════════════════════════════════════╣
║  ĐÈ PHÍM = DI CHUYỂN  |  BỎ RA = DỪNG     ║
╚══════════════════════════════════════════════╝
"""


def main(args=None):
    rclpy.init(args=args)
    node = Node('keyboard_teleop')
    pub  = node.create_publisher(Twist, '/cmd_vel', 10)

    print(HELP)
    print(f"  MAX_V={MAX_V} m/s  |  MAX_W={MAX_W} rad/s  |  Timeout={TIMEOUT_S*1000:.0f}ms\n")

    last_key  = ''
    was_moving = False

    def send(v, w):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        pub.publish(msg)

    try:
        while rclpy.ok():
            key = get_key(TIMEOUT_S)

            if key == 'q':
                print("\nThoát.")
                break

            if key in KEY_MAP:
                v, w = KEY_MAP[key]
                send(v, w)

                if key != last_key or not was_moving:
                    direction = {
                        'w': 'Tiến',
                        's': 'Lùi',
                        'a': 'Quay trái',
                        'd': 'Quay phải',
                    }
                    print(f"\r  [{direction[key]}]  v={v:+.2f} m/s  w={w:+.2f} rad/s    ",
                          end='', flush=True)
                last_key   = key
                was_moving = True

            else:
                # Timeout → không có phím → dừng robot
                if was_moving:
                    send(0.0, 0.0)
                    print(f"\r  [DỪNG]                                          ",
                          end='', flush=True)
                    was_moving = False
                last_key = ''

            rclpy.spin_once(node, timeout_sec=0)

    except KeyboardInterrupt:
        pass
    finally:
        send(0.0, 0.0)
        print("\n  Robot đã dừng. Node thoát.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

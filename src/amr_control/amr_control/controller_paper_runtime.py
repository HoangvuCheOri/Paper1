#!/usr/bin/env python3
"""Shared one-node runtime for BSMC/Backstepping paper experiments."""

import math

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException

from amr_control.embedded_paper_capture import EmbeddedPaperCapture


def expected_eight_alignment_duration(node):
    """Estimate the bounded rotate-in-place phase instead of using timeout."""
    start_phase = float(getattr(node, "START_PHASE", 0.0))
    local_tangent = math.atan2(math.cos(2.0 * start_phase), math.cos(start_phase))
    target = (
        math.radians(float(getattr(node, "PATH_ROTATION_DEG", 0.0)))
        + local_tangent
    )
    current = float(getattr(node, "current_theta", 0.0))
    error = abs(math.atan2(math.sin(target - current), math.cos(target - current)))
    kp = max(float(getattr(node, "INITIAL_ALIGN_KP", 1.2)), 1e-6)
    max_w = max(float(getattr(node, "INITIAL_ALIGN_MAX_W", 0.35)), 1e-6)
    tolerance = max(float(getattr(node, "INITIAL_ALIGN_TOLERANCE", math.radians(3.0))), 1e-6)
    hold = max(float(getattr(node, "INITIAL_ALIGN_HOLD_TIME", 0.3)), 0.0)
    linear_threshold = max_w / kp
    saturated_time = max(0.0, error - linear_threshold) / max_w
    linear_start = min(error, linear_threshold)
    linear_time = (
        math.log(linear_start / tolerance) / kp
        if linear_start > tolerance else 0.0
    )
    model_estimate = saturated_time + linear_time + hold + 0.75
    hardware_estimate = max(
        float(getattr(node, "INITIAL_ALIGN_DURATION_ESTIMATE", 0.0)), 0.0
    )
    estimate = max(model_estimate, hardware_estimate)
    minimum = max(float(getattr(node, "INITIAL_ALIGN_TIME", 0.0)), 0.0)
    timeout = max(float(getattr(node, "INITIAL_ALIGN_TIMEOUT", estimate)), minimum)
    return min(timeout, max(minimum, estimate))


def automatic_duration(node, trajectory, laps=1.0):
    """Return wall time for the requested number of complete trajectories."""
    laps = max(1.0, float(laps))
    startup = max(0.0, float(getattr(node, "STARTUP_DELAY", 1.0)))
    settling = max(0.0, float(getattr(node, "SETTLE_TIME", 2.0)))
    stop_buffer = 2.0
    if trajectory == "circle":
        # During a linear velocity ramp the accumulated phase loses half the
        # ramp duration. Compensate exactly, and do not keep driving through
        # the generic two-second export buffer (which caused a 10.74 cm
        # reference endpoint gap in run 20260719_181813).
        motion = (
            laps * 2.0 * math.pi / max(abs(float(node.W)), 1e-6)
            + 0.5 * float(getattr(node, "TRAJECTORY_RAMP_TIME", 2.0))
        )
        stop_buffer = 0.0
    elif trajectory == "eight":
        # bsmc_eight ramps the phase rate linearly. During that interval the
        # integrated phase is W*T_ramp/2 rather than W*T_ramp, so reaching a
        # complete 2*pi cycle needs an extra T_ramp/2 of wall time.
        motion = (
            laps * float(node.T_period)
            + 0.5 * float(getattr(node, "TRAJECTORY_RAMP_TIME", 0.0))
        )
        stop_buffer = 0.0
        if not bool(getattr(node, "initial_alignment_complete", True)):
            motion += expected_eight_alignment_duration(node)
    elif trajectory == "square":
        # This is only an informational estimate. EmbeddedPaperCapture stops
        # Square from measured path_progress, because corner deceleration
        # makes a clock-only lap count inaccurate.
        motion = laps * float(node.loop_length) / max(float(node.VD), 1e-6)
    else:
        raise ValueError(f"Unsupported trajectory: {trajectory}")
    return startup + settling + motion + stop_buffer


def run_controller(node_factory, controller_label, trajectory, configure=None, args=None):
    """Spin one controller with logging/plots embedded in that same node."""
    rclpy.init(args=args)
    node = node_factory()
    capture = None
    try:
        if configure is not None:
            configure(node)
        node.declare_parameter(
            "paper_laps",
            1.0,
            ParameterDescriptor(dynamic_typing=True),
        )
        laps = max(1.0, float(node.get_parameter("paper_laps").value))
        duration = automatic_duration(node, trajectory, laps=laps)
        capture = EmbeddedPaperCapture(
            node,
            controller=controller_label,
            trajectory=trajectory,
            default_duration=duration,
        )
        node.get_logger().info(
            f"Paper mode: controller={controller_label}, trajectory={trajectory}, "
            f"laps={laps:g}. "
            "Use paper_duration:=0 for manual Ctrl-C."
        )
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.cmd_pub.publish(Twist())
        finally:
            if capture is not None:
                capture.finalize()
            node.destroy_node()
            rclpy.try_shutdown()


def force_backstepping(node):
    """Remove only the sliding-mode injection; retain identical trajectory logic."""
    node.Ks1 = 0.0
    node.Ks2 = 0.0
    node.get_logger().info(
        "Backstepping baseline active: Ks1=Ks2=0; trajectory and all other "
        "parameters are shared with the corresponding BSMC implementation."
    )

#!/usr/bin/env python3
"""Paper-enabled entry points for the unchanged BSMC controllers."""

from amr_control.bsmc_circle import BSMCCircle
from amr_control.bsmc_eight import BSMCEight
from amr_control.bsmc_square import BSMCSquare
from amr_control.controller_paper_runtime import run_controller


def _validated_circle_bsmc(node):
    # Validated compensated run 20260718_184213.  The historical controller
    # default has Ks1=Ks2=0 and therefore belongs to the Backstepping baseline.
    if node.Ks1 == 0.0 and node.Ks2 == 0.0:
        node.Ks1 = 0.024
        node.Ks2 = 0.050
        node.get_logger().info(
            "Circle BSMC validated sliding gains active: Ks1=0.024, Ks2=0.050."
        )


def circle_main(args=None):
    run_controller(
        BSMCCircle, "BSMC", "circle",
        configure=_validated_circle_bsmc, args=args,
    )


def eight_main(args=None):
    run_controller(BSMCEight, "BSMC", "eight", args=args)


def square_main(args=None):
    run_controller(BSMCSquare, "BSMC", "square", args=args)

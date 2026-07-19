#!/usr/bin/env python3
"""Circle Backstepping baseline with embedded paper capture."""

from amr_control.bsmc_circle import BSMCCircle
from amr_control.controller_paper_runtime import force_backstepping, run_controller


def main(args=None):
    run_controller(
        BSMCCircle, "Backstepping", "circle",
        configure=force_backstepping, args=args,
    )


if __name__ == "__main__":
    main()

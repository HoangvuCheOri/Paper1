#!/usr/bin/env python3
"""Figure-eight Backstepping baseline with embedded paper capture."""

from amr_control.bsmc_eight import BSMCEight
from amr_control.controller_paper_runtime import force_backstepping, run_controller


def main(args=None):
    run_controller(
        BSMCEight, "Backstepping", "eight",
        configure=force_backstepping, args=args,
    )


if __name__ == "__main__":
    main()

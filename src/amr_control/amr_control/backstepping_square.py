#!/usr/bin/env python3
"""Square Backstepping baseline with embedded paper capture."""

from amr_control.bsmc_square import BSMCSquare
from amr_control.controller_paper_runtime import force_backstepping, run_controller


def main(args=None):
    run_controller(
        BSMCSquare, "Backstepping", "square",
        configure=force_backstepping, args=args,
    )


if __name__ == "__main__":
    main()

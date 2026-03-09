"""
Thin entry point for the MemPrO console script.

JAX reads NUM_CPU and MEMPRO_PLATFORM from the environment at import time to
configure its device mesh, so these must be set *before* mempro.MemPrO is
imported.  A normal console_scripts entry point would call main() after the
module is already imported, so this shim pre-parses just those two flags,
sets the env vars, and only then imports and calls the real main().
"""

import os
import argparse


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("-nc", "--num_cpus", type=int, default=None)
    _pre.add_argument("-p", "--platform", default=None, choices=["cpu", "gpu"])
    _pre_args, _ = _pre.parse_known_args()

    if _pre_args.num_cpus is not None:
        os.environ["NUM_CPU"] = str(_pre_args.num_cpus)
    if _pre_args.platform is not None:
        os.environ["MEMPRO_PLATFORM"] = _pre_args.platform

    # Import is deliberately deferred until after env vars are set above,
    # so that JAX initialises with the correct platform and device count.
    from mempro import MemPrO
    MemPrO.main()

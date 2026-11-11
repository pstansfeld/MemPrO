"""
Entry points for the mempro and insane4mempro console scripts.

Device options (-nc, -dev) are read when MemPrO is imported, because JAX has
to be configured before it starts; see _pre_parse_device_args in MemPrO.py.
"""


def main():
    from mempro import MemPrO
    MemPrO.main()


def insane():
    """Runs Insane4MemPrO.py, which is a script rather than a module, with the command line it was given."""
    import os
    import runpy
    import sys
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Insane4MemPrO.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")

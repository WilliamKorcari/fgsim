"""Main module."""
import sys

import pretty_errors

pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
    code_color="  " + pretty_errors.default_config.line_color,
    truncate_code=True,
    display_locals=True,
)


def main():
    # always reload the local modules
    # so that
    # `ipython >>> %run -m fgsim train`
    # works
    for modulename in [
        e for e in sys.modules if e.startswith("fgsim.") and "mapper" not in e
    ]:
        del sys.modules[modulename]
    from .cli import args

    # if args.command == "geo":
    #     from .geo import mapper
    #     _ = mapper.geomapper("data/test.toml")
    #     import numpy as np
    #     import geomapper as xt

    if args.command == "train":
        from .train.holder import modelHolder
        from .train.train import training_procedure

        m = modelHolder()
        training_procedure(m)

    if args.command == "generate":
        from .train.generate import generation_procedure
        from .train.holder import modelHolder

        m = modelHolder()
        generation_procedure(m)


if __name__ == "__main__":
    main()

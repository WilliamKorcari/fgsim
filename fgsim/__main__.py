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
    local_package_name = "fgsim"
    local_modules = {e for e in sys.modules if e.startswith(local_package_name)}
    do_not_reload = {
        # Never remove the upper packages
        "fgsim",
        "fgsim.geo",
        "fgsim.train",
        # Always reload cli and config
        # "fgsim.cli",
        # "fgsim.config",
        # utils dont change frequently
        "fgsim.utils",
        "fgsim.plot",
        # The rest
        "fgsim.geo.mapper",
        "fgsim.train.train",
        "fgsim.train.model",
        "fgsim.train.generate",
        "fgsim.data_loader",
        "fgsim.train.holder",
        # Currently working on:
        # "fgsim.data_dumper",
        # "fgsim.geo.mapback",
    }
    for modulename in local_modules - do_not_reload:
        print(f"Unloading {modulename}")
        del sys.modules[modulename]
    print("Unloading complete")
    from .cli import args

    # if args.command == "geo":
    #     from .geo import mapper
    #     _ = mapper.geomapper("data/test.toml")
    #     import numpy as np
    #     import geomapper as xt

    if args.command == "train":
        from .train.holder import model_holder
        from .train.train import training_procedure

        training_procedure(model_holder)

    if args.command == "generate":
        from .train.generate import generation_procedure
        from .train.holder import model_holder

        generation_procedure(model_holder)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
CryoAtom: Automated Cryo-EM model building toolkit
"""


def main():
    import argparse
    import warnings

    import CryoAtom

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CryoAtom {CryoAtom.__version__}",
    )

    # Suppress some warnings
    warnings.filterwarnings("ignore", message="^.* socket cannot be initialized.*$")

    import CryoAtom.build
    import CryoAtom.AlignWithTemplate

    modules = {
        "build": CryoAtom.build,
        "assemble": CryoAtom.AlignWithTemplate,
    }

    subparsers = parser.add_subparsers(title="Choose a module",)
    subparsers.required = "True"

    for key in modules:
        module_parser = subparsers.add_parser(
            key,
            description=modules[key].__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )
        modules[key].add_args(module_parser)
        module_parser.set_defaults(func=modules[key].main)

    try:
        args = parser.parse_args()
        args.func(args)
    except TypeError:
        parser.print_help()


if __name__ == "__main__":
    main()

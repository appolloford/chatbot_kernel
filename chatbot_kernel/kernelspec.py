import argparse
import json
import os
import sys
import shutil

from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager
from tempfile import TemporaryDirectory

kernel_json = {
    "argv": [sys.executable, "-m", "chatbot_kernel", "-f", "{connection_file}"],
    "display_name": "Chatbot",
    "language": "text",
}


def install_chatbot_kernel_spec(user=True, prefix=None):
    with TemporaryDirectory() as td:
        os.chmod(td, 0o755)  # Starts off as 700, not user readable
        with open(os.path.join(td, "kernel.json"), "w") as f:
            json.dump(kernel_json, f, sort_keys=True)

        print("Installing Chatbot Kernel spec")
        # Requires logo files in kernel root directory
        cur_path = os.path.dirname(os.path.realpath(__file__))
        for logo in ["logo-32x32.png", "logo-64x64.png"]:
            try:
                shutil.copy(os.path.join(cur_path, logo), td)
            except FileNotFoundError:
                print("Custom logo files not found. Default logos will be used.")

        KernelSpecManager().install_kernel_spec(td, "chatbot", user=user, prefix=prefix)


def _is_root():
    try:
        return os.geteuid() == 0
    except AttributeError:
        return False  # assume not an admin on non-Unix platforms


def main(argv=None):
    parser = argparse.ArgumentParser(description="Install KernelSpec for Chatbot Kernel")
    prefix_locations = parser.add_mutually_exclusive_group()

    prefix_locations.add_argument(
        "--user",
        help="Install KernelSpec in user home directory",
        action="store_true",
    )
    prefix_locations.add_argument(
        "--sys-prefix",
        help="Install KernelSpec in sys.prefix. Useful in conda / virtualenv",
        action="store_true",
        dest="sys_prefix",
    )
    prefix_locations.add_argument(
        "--prefix",
        help="Install KernelSpec in this prefix",
        default=None,
    )

    args = parser.parse_args(argv)

    user = False
    prefix = None
    if args.sys_prefix:
        prefix = sys.prefix
    elif args.prefix:
        prefix = args.prefix
    elif args.user or not _is_root():
        user = True

    install_chatbot_kernel_spec(user=user, prefix=prefix)


if __name__ == "__main__":
    main()

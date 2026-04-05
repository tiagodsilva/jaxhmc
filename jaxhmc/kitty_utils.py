import os
import subprocess
from io import BytesIO

import matplotlib.pyplot as plt


def is_kitty():
    term = os.environ.get("TERM", "")
    # Both terminals support graphics
    is_kitty = term.startswith("xterm-kitty")
    is_ghostty = term.startswith("xterm-ghostty")
    return is_kitty or is_ghostty


def show_kitty():
    # Utility for displaying plots in kitty terminal
    # (instead of creating a window)
    if not is_kitty():
        return

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Pipe directly to kitty
    subprocess.run(["kitty", "+kitten", "icat"], input=buf.getvalue())
    plt.close()

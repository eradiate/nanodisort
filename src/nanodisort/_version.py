# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib.metadata import PackageNotFoundError, version

try:
    _version = version("nanodisort")
except PackageNotFoundError as e:
    raise PackageNotFoundError(
        "Eradiate is not installed; please install it in your Python environment."
    ) from e

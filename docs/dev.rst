Development
===========

Prerequisites
-------------

* Linux or macOS
* C compiler (gcc, clang)
* CMake 3.15 or later
* `uv <https://github.com/astral-sh/uv>`__
* Python 3.9 to 3.13
* NumPy 1.20 or later

Quick instructions
------------------

Clone the repository
    .. code-block:: bash

        git clone https://github.com/rayference/nanodisort.git
        cd nanodisort

Setup development environment
    .. code-block:: bash

        uv sync --dev
        uv pip install -ve .

(Re)build C++ module
    .. code-block:: bash

        uv pip install -ve.

Run tests
    .. code-block:: bash

        uv run task test

Run linting
    .. code-block:: bash

        uv run task lint

Format code
    .. code-block:: bash

        uv run task format


Build documentation
    .. code-block:: bash

        uv run task docs  # static build
        uv run task docs-serve  # server with auto-rebuild

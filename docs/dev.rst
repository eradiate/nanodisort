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

Run benchmarks
    .. code-block:: bash

        uv run task benchmark

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

Release sequence
----------------

1. Make sure all CI checks pass.
2. Set the ``$RELEASE_VERSION`` environment variable to the target value:

   .. code-block:: bash

       export RELEASE_VERSION=X.Y.Z

3. Bump the version to the target value:

   .. code-block:: bash

       uv version $RELEASE_VERSION

4. Update the release notes (CHANGELOG.md).
5. Commit and push the changes:

   .. code-block:: bash

       git commit -am "Bump version to $RELEASE_VERSION"
       git push origin main

6. Create a tag for the target release and push it:

   .. code-block:: bash

       git tag v$RELEASE_VERSION
       git push origin v$RELEASE_VERSION

   This will automatically run the GitHub Actions build job and publish the package to PyPI. Proceed with care!

7. Bump the version to the next development version:

   .. code-block:: bash

       uv version --bump minor --bump dev

   Use the ``--dry-run`` flag in case of doubt.

8. Commit and push the changes:

   .. code-block:: bash

       git commit -am "Ready for next development cycle"
       git push origin main

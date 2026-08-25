# nanodisort — changelog

---

## v0.3.0 (25th August 2026)

* Added bindings for `radius` and `zd`, required to use the pseudo-spherical
  correction. The `spher` flag was previously bound without them, which made it
  silently produce a degenerate geometry.
* Built with nanobind 3.0 in split mode: the nanobind runtime now lives in the
  separate `nanobind-backend` package, and nanodisort ships a single abi3 wheel
  per platform instead of one per Python version.
* Dropped support for Python 3.9 (required by nanobind 3.0).
* Added support for Python 3.14.

## v0.2.0 (2nd June 2026)

* Added bindings for `nphase`, `phase` and `mu_phase`, required to use the
  Buras-Emde intensity correction.
* Added Buras-Emde intensity correction.
* Added `disotest` test cases 9 and 10.

## v0.1.0 (13th April 2026)

*Initial release.*

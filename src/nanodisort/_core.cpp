/*
 * nanodisort - Python bindings for cdisort
 *
 * Copyright (c) 2025 Rayference
 * Licensed under GPL-3.0-or-later
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>

extern "C" {
#include "cdisort.h"
}

namespace nb = nanobind;
using namespace nb::literals;

// Type aliases for numpy arrays
using ArrayD1 = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrayD2 = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/*
 * DisortState class - wraps cdisort's disort_state structure
 *
 * This class manages the lifecycle of the disort_state structure and provides
 * a Pythonic interface to the cdisort solver.
 */
class DisortState {
public:
    disort_state ds;
    disort_output out;
    bool allocated;

    DisortState() : allocated(false) {
        // Initialize structure to zeros
        memset(&ds, 0, sizeof(disort_state));
        memset(&out, 0, sizeof(disort_output));
    }

    ~DisortState() {
        if (allocated) {
            c_disort_out_free(&ds, &out);
            c_disort_state_free(&ds);
        }
    }

    // Disable copy (move-only type)
    DisortState(const DisortState&) = delete;
    DisortState& operator=(const DisortState&) = delete;

    /*
     * Allocate memory for arrays based on configured dimensions
     * Must be called after setting all dimensions and flags
     */
    void allocate() {
        if (allocated) {
            c_disort_out_free(&ds, &out);
            c_disort_state_free(&ds);
        }
        c_disort_state_alloc(&ds);
        c_disort_out_alloc(&ds, &out);
        allocated = true;
    }

    /*
     * Run the DISORT solver
     */
    void solve() {
        if (!allocated) {
            throw std::runtime_error("Memory not allocated. Call allocate() first.");
        }
        c_disort(&ds, &out);
    }

    // Dimension setters/getters
    void set_nstr(int nstr) { ds.nstr = nstr; }
    int get_nstr() const { return ds.nstr; }

    void set_nlyr(int nlyr) { ds.nlyr = nlyr; }
    int get_nlyr() const { return ds.nlyr; }

    void set_nmom(int nmom) { ds.nmom = nmom; }
    int get_nmom() const { return ds.nmom; }

    void set_ntau(int ntau) { ds.ntau = ntau; }
    int get_ntau() const { return ds.ntau; }

    void set_numu(int numu) { ds.numu = numu; }
    int get_numu() const { return ds.numu; }

    void set_nphi(int nphi) { ds.nphi = nphi; }
    int get_nphi() const { return ds.nphi; }

    /*
     * String representation showing key dimensions and allocation status
     */
    std::string repr() const {
        std::ostringstream oss;
        oss << "<DisortState"
            << " nstr=" << ds.nstr
            << " nlyr=" << ds.nlyr
            << " nmom=" << ds.nmom
            << " ntau=" << ds.ntau
            << " numu=" << ds.numu
            << " nphi=" << ds.nphi
            << " allocated=" << (allocated ? "True" : "False")
            << ">";
        return oss.str();
    }

    // TODO: Add methods for:
    // - Setting optical properties (dtauc, ssalb, pmom)
    // - Setting geometry (umu, phi, utau)
    // - Setting boundary conditions
    // - Getting output arrays
};

/*
 * Python module definition
 */
NB_MODULE(_core, m) {
    m.doc() = "nanodisort core bindings to cdisort";

    // DisortState class
    nb::class_<DisortState>(m, "DisortState", "DISORT solver state")
        .def(nb::init<>(), "Create a new DISORT state")
        .def("__repr__", &DisortState::repr)
        .def("allocate", &DisortState::allocate,
             "Allocate memory for arrays based on configured dimensions")
        .def("solve", &DisortState::solve,
             "Run the DISORT solver")
        // Dimensions
        .def_prop_rw("nstr", &DisortState::get_nstr, &DisortState::set_nstr,
                     "Number of streams")
        .def_prop_rw("nlyr", &DisortState::get_nlyr, &DisortState::set_nlyr,
                     "Number of computational layers")
        .def_prop_rw("nmom", &DisortState::get_nmom, &DisortState::set_nmom,
                     "Number of phase function moments")
        .def_prop_rw("ntau", &DisortState::get_ntau, &DisortState::set_ntau,
                     "Number of user optical depths")
        .def_prop_rw("numu", &DisortState::get_numu, &DisortState::set_numu,
                     "Number of user polar angles")
        .def_prop_rw("nphi", &DisortState::get_nphi, &DisortState::set_nphi,
                     "Number of user azimuthal angles");
}

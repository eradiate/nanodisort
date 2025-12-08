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
#include <algorithm>
#include <sstream>

extern "C" {
#include "cdisort.h"
}

namespace nb = nanobind;
using namespace nb::literals;

// Type aliases for numpy arrays
using ArrayD1 = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrayD2 = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Error message constants
constexpr const char* NOT_ALLOCATED_ERROR = "Memory not allocated. Call allocate() first.";
constexpr const char* OUTPUT_UNAVAILABLE_ERROR = "Output not available. Run solve() first.";

/*
 * Error handling for cdisort
 * Callback throws C++ exception that will be caught by nanobind
 */

// Error callback that will be called by cdisort instead of exit()
// This function throws a C++ exception which will be converted to a Python exception by nanobind
extern "C" void cdisort_error_handler(const char* message) {
    throw std::runtime_error(std::string("DISORT error: ") + message);
}

// Macros for property getters/setters to reduce boilerplate

// Integer dimension properties
#define DEFINE_INT_PROPERTY(name, member) \
    void set_##name(int value) { member = value; } \
    int get_##name() const { return member; }

// Flag properties (bool <-> TRUE/FALSE conversion)
#define DEFINE_FLAG_PROPERTY(name, member) \
    void set_##name(bool val) { member = val ? TRUE : FALSE; } \
    bool get_##name() const { return member == TRUE; }

// Scalar double properties
#define DEFINE_SCALAR_PROPERTY(name, member) \
    void set_##name(double value) { member = value; } \
    double get_##name() const { return member; }

// 1D array properties with size checking
#define DEFINE_ARRAY_PROPERTY(name, member, size_expr) \
    void set_##name(ArrayD1 arr) { \
        check_allocated(); \
        check_array_size(arr, size_expr, #name); \
        std::copy_n(arr.data(), size_expr, member); \
    } \
    auto get_##name() { \
        check_allocated(); \
        size_t shape[1] = {static_cast<size_t>(size_expr)}; \
        return nb::ndarray<nb::numpy, double, nb::ndim<1>>(member, 1, shape); \
    }

// Output array property (read-only, creates owned copy)
#define DEFINE_OUTPUT_PROPERTY(name, field) \
    auto get_##name() { \
        check_allocated(); \
        if (!out.rad) { \
            throw std::runtime_error(OUTPUT_UNAVAILABLE_ERROR); \
        } \
        double* data = new double[ds.ntau]; \
        for (int i = 0; i < ds.ntau; i++) { \
            data[i] = out.rad[i].field; \
        } \
        size_t shape[1] = {static_cast<size_t>(ds.ntau)}; \
        nb::capsule owner(data, [](void *p) noexcept { \
            delete[] static_cast<double*>(p); \
        }); \
        return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, 1, shape, owner); \
    }

// Nanobind property binding macros
#define BIND_PROPERTY_RW(name, doc) \
    .def_prop_rw(#name, &DisortState::get_##name, &DisortState::set_##name, doc)

#define BIND_PROPERTY_RO(name, doc) \
    .def_prop_ro(#name, &DisortState::get_##name, nb::rv_policy::move, doc)

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

        // Register error callback (do this once, static)
        static bool callback_registered = false;
        if (!callback_registered) {
            c_set_error_callback(cdisort_error_handler);
            callback_registered = true;
        }
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
    DEFINE_INT_PROPERTY(nstr, ds.nstr)
    DEFINE_INT_PROPERTY(nlyr, ds.nlyr)
    DEFINE_INT_PROPERTY(nmom, ds.nmom)
    DEFINE_INT_PROPERTY(ntau, ds.ntau)
    DEFINE_INT_PROPERTY(numu, ds.numu)
    DEFINE_INT_PROPERTY(nphi, ds.nphi)

    // Control flags
    DEFINE_FLAG_PROPERTY(usrtau, ds.flag.usrtau)
    DEFINE_FLAG_PROPERTY(usrang, ds.flag.usrang)
    DEFINE_FLAG_PROPERTY(lamber, ds.flag.lamber)
    DEFINE_FLAG_PROPERTY(planck, ds.flag.planck)
    DEFINE_FLAG_PROPERTY(onlyfl, ds.flag.onlyfl)
    DEFINE_FLAG_PROPERTY(quiet, ds.flag.quiet)
    DEFINE_FLAG_PROPERTY(intensity_correction, ds.flag.intensity_correction)
    DEFINE_FLAG_PROPERTY(spher, ds.flag.spher)

    // Boundary conditions
    DEFINE_SCALAR_PROPERTY(fbeam, ds.bc.fbeam)
    DEFINE_SCALAR_PROPERTY(umu0, ds.bc.umu0)
    DEFINE_SCALAR_PROPERTY(phi0, ds.bc.phi0)
    DEFINE_SCALAR_PROPERTY(fisot, ds.bc.fisot)
    DEFINE_SCALAR_PROPERTY(fluor, ds.bc.fluor)
    DEFINE_SCALAR_PROPERTY(albedo, ds.bc.albedo)
    DEFINE_SCALAR_PROPERTY(btemp, ds.bc.btemp)
    DEFINE_SCALAR_PROPERTY(ttemp, ds.bc.ttemp)
    DEFINE_SCALAR_PROPERTY(temis, ds.bc.temis)

    // Other scalar parameters
    DEFINE_SCALAR_PROPERTY(accur, ds.accur)
    DEFINE_SCALAR_PROPERTY(wvnmlo, ds.wvnmlo)
    DEFINE_SCALAR_PROPERTY(wvnmhi, ds.wvnmhi)

    // Array setters/getters - require allocation first
    DEFINE_ARRAY_PROPERTY(dtauc, ds.dtauc, ds.nlyr)
    DEFINE_ARRAY_PROPERTY(ssalb, ds.ssalb, ds.nlyr)

    void set_pmom(ArrayD2 arr) {
        check_allocated();
        int expected_size = (ds.nmom_nstr + 1) * ds.nlyr;
        if (arr.shape(0) != ds.nmom_nstr + 1 || arr.shape(1) != ds.nlyr) {
            throw std::runtime_error(
                "pmom array shape mismatch. Expected ("
                + std::to_string(ds.nmom_nstr + 1) + ", "
                + std::to_string(ds.nlyr) + ")"
            );
        }
        std::copy_n(arr.data(), expected_size, ds.pmom);
    }

    auto get_pmom() {
        check_allocated();
        size_t shape[2] = {
            static_cast<size_t>(ds.nmom_nstr + 1),
            static_cast<size_t>(ds.nlyr)
        };
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            ds.pmom, 2, shape
        );
    }

    DEFINE_ARRAY_PROPERTY(umu, ds.umu, ds.numu)
    DEFINE_ARRAY_PROPERTY(phi, ds.phi, ds.nphi)
    DEFINE_ARRAY_PROPERTY(utau, ds.utau, ds.ntau)
    DEFINE_ARRAY_PROPERTY(temper, ds.temper, ds.nlyr + 1)

    // Output accessors (read-only) - create copies owned by Python
    DEFINE_OUTPUT_PROPERTY(rfldir, rfldir)
    DEFINE_OUTPUT_PROPERTY(rfldn, rfldn)
    DEFINE_OUTPUT_PROPERTY(flup, flup)
    DEFINE_OUTPUT_PROPERTY(dfdt, dfdt)
    DEFINE_OUTPUT_PROPERTY(uavg, uavg)
    DEFINE_OUTPUT_PROPERTY(uavgdn, uavgdn)
    DEFINE_OUTPUT_PROPERTY(uavgup, uavgup)
    DEFINE_OUTPUT_PROPERTY(uavgso, uavgso)

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

private:
    void check_allocated() const {
        if (!allocated) {
            throw std::runtime_error(NOT_ALLOCATED_ERROR);
        }
    }

    void check_array_size(const ArrayD1& arr, int expected, const char* name) const {
        if (arr.shape(0) != expected) {
            throw std::runtime_error(
                std::string(name) + " array size mismatch. Expected "
                + std::to_string(expected) + ", got " + std::to_string(arr.shape(0))
            );
        }
    }
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
        BIND_PROPERTY_RW(nstr, "Number of streams")
        BIND_PROPERTY_RW(nlyr, "Number of computational layers")
        BIND_PROPERTY_RW(nmom, "Number of phase function moments")
        BIND_PROPERTY_RW(ntau, "Number of user optical depths")
        BIND_PROPERTY_RW(numu, "Number of user polar angles")
        BIND_PROPERTY_RW(nphi, "Number of user azimuthal angles")
        // Control flags
        BIND_PROPERTY_RW(usrtau, "Return radiant quantities at user-specified optical depths")
        BIND_PROPERTY_RW(usrang, "Return radiant quantities at user-specified polar angles")
        BIND_PROPERTY_RW(lamber, "Isotropically reflecting bottom boundary")
        BIND_PROPERTY_RW(planck, "Include thermal emission")
        BIND_PROPERTY_RW(onlyfl, "Return only fluxes (not intensities)")
        BIND_PROPERTY_RW(quiet, "Suppress output messages")
        BIND_PROPERTY_RW(intensity_correction, "Apply intensity correction")
        BIND_PROPERTY_RW(spher, "Pseudo-spherical geometry (vs plane-parallel)")
        // Boundary conditions
        BIND_PROPERTY_RW(fbeam, "Intensity of incident parallel beam")
        BIND_PROPERTY_RW(umu0, "Polar angle cosine of incident beam")
        BIND_PROPERTY_RW(phi0, "Azimuth angle of incident beam (degrees)")
        BIND_PROPERTY_RW(fisot, "Intensity of top-boundary isotropic illumination")
        BIND_PROPERTY_RW(fluor, "Intensity of bottom-boundary isotropic illumination")
        BIND_PROPERTY_RW(albedo, "Albedo of bottom boundary")
        BIND_PROPERTY_RW(btemp, "Temperature [K] of bottom boundary")
        BIND_PROPERTY_RW(ttemp, "Temperature [K] of top boundary")
        BIND_PROPERTY_RW(temis, "Emissivity of top boundary")
        // Other scalar parameters
        BIND_PROPERTY_RW(accur, "Convergence criterion for azimuthal series")
        BIND_PROPERTY_RW(wvnmlo, "Wavenumber lower bound [cm^-1] for Planck function")
        BIND_PROPERTY_RW(wvnmhi, "Wavenumber upper bound [cm^-1] for Planck function")
        // Optical property arrays
        BIND_PROPERTY_RW(dtauc, "Optical depths of computational layers [nlyr]")
        BIND_PROPERTY_RW(ssalb, "Single-scatter albedos [nlyr]")
        BIND_PROPERTY_RW(pmom, "Phase function moments [nmom_nstr+1, nlyr]")
        // Geometry arrays
        BIND_PROPERTY_RW(umu, "Polar angle cosines [numu]")
        BIND_PROPERTY_RW(phi, "Azimuthal angles [degrees] [nphi]")
        BIND_PROPERTY_RW(utau, "User optical depths [ntau]")
        BIND_PROPERTY_RW(temper, "Temperatures [K] at levels [nlyr+1]")
        // Output arrays (read-only)
        BIND_PROPERTY_RO(rfldir, "Direct beam flux (without delta-M scaling) [ntau]")
        BIND_PROPERTY_RO(rfldn, "Diffuse downward flux (without delta-M scaling) [ntau]")
        BIND_PROPERTY_RO(flup, "Diffuse upward flux [ntau]")
        BIND_PROPERTY_RO(dfdt, "Flux divergence d(net flux)/d(optical depth) [ntau]")
        BIND_PROPERTY_RO(uavg, "Mean intensity including direct beam [ntau]")
        BIND_PROPERTY_RO(uavgdn, "Mean diffuse downward intensity [ntau]")
        BIND_PROPERTY_RO(uavgup, "Mean diffuse upward intensity [ntau]")
        BIND_PROPERTY_RO(uavgso, "Mean direct solar intensity [ntau]");
}

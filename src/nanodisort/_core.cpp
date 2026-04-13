// SPDX-FileCopyrightText: 2025 Rayference
//
// SPDX-License-Identifier: GPL-3.0-or-later

/*
 * nanodisort - Python bindings for CDISORT
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <algorithm>
#include <future>
#include <memory>
#include <sstream>
#include <vector>

#include "thread_pool.h"

extern "C" {
#include "cdisort.h"
}

namespace nb = nanobind;
using namespace nb::literals;

// Type aliases for numpy arrays
using ArrayD1 = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrayD2 = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ArrayD3 = nb::ndarray<double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
// Fortran-contiguous 2D arrays (cdisort uses column-major order internally)
using ArrayD2F = nb::ndarray<double, nb::ndim<2>, nb::f_contig, nb::device::cpu>;
using ArrayD3F = nb::ndarray<double, nb::ndim<3>, nb::f_contig, nb::device::cpu>;

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

// Helper to make a numpy array read-only via setflags(write=False)
template<typename T>
nb::object make_readonly(T&& arr) {
    nb::object obj = nb::cast(std::forward<T>(arr));
    obj.attr("setflags")("write"_a = false);
    return obj;
}

// Output array properties for radiant quantities (stored in out.rad[].field)
// Returns a read-only copy.
#define DEFINE_OUTPUT_RADIANT(name, field) \
    nb::object get_##name() { \
        check_allocated(); \
        if (!out.rad) { \
            throw std::runtime_error("Output not available. Run solve() first."); \
        } \
        double* data = new double[ds.ntau]; \
        for (int i = 0; i < ds.ntau; i++) { \
            data[i] = out.rad[i].field; \
        } \
        size_t shape[1] = {static_cast<size_t>(ds.ntau)}; \
        nb::capsule owner(data, [](void *p) noexcept { \
            delete[] static_cast<double*>(p); \
        }); \
        return make_readonly(nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, 1, shape, owner)); \
    }

// Generic output array property for 1D arrays. Returns a read-only copy.
#define DEFINE_OUTPUT_ARRAY_1D(name, ptr, size_expr, err_msg) \
    nb::object get_##name() { \
        check_allocated(); \
        if (!out.ptr) { \
            throw std::runtime_error(err_msg); \
        } \
        int total_size = size_expr; \
        double* data = new double[total_size]; \
        std::copy_n(out.ptr, total_size, data); \
        size_t shape[1] = {static_cast<size_t>(total_size)}; \
        nb::capsule owner(data, [](void *p) noexcept { \
            delete[] static_cast<double*>(p); \
        }); \
        return make_readonly(nb::ndarray<nb::numpy, double, nb::ndim<1>>(data, 1, shape, owner)); \
    }

// Note: cdisort stores 2D arrays in Fortran (column-major) order.
// We return a read-only view with Fortran strides; caller must keep DisortState alive.
#define DEFINE_OUTPUT_ARRAY_2D(name, ptr, dim1, dim2, err_msg) \
    nb::object get_##name() { \
        check_allocated(); \
        if (!out.ptr) { \
            throw std::runtime_error(err_msg); \
        } \
        int d1 = dim1, d2 = dim2; \
        size_t shape[2] = {static_cast<size_t>(d1), static_cast<size_t>(d2)}; \
        /* Fortran strides in elements */ \
        int64_t strides[2] = {1, static_cast<int64_t>(d1)}; \
        return make_readonly(nb::ndarray<nb::numpy, double, nb::ndim<2>>( \
            out.ptr, 2, shape, nb::handle(), strides \
        )); \
    }

// Note: cdisort stores multidimensional arrays in Fortran (column-major) order.
// We return a read-only view with Fortran strides; caller must keep DisortState alive.
#define DEFINE_OUTPUT_ARRAY_3D(name, ptr, dim1, dim2, dim3, err_msg) \
    nb::object get_##name() { \
        check_allocated(); \
        if (!out.ptr) { \
            throw std::runtime_error(err_msg); \
        } \
        int d1 = dim1, d2 = dim2, d3 = dim3; \
        size_t shape[3] = { \
            static_cast<size_t>(d1), \
            static_cast<size_t>(d2), \
            static_cast<size_t>(d3) \
        }; \
        /* Fortran strides in elements */ \
        int64_t strides[3] = { \
            1, \
            static_cast<int64_t>(d1), \
            static_cast<int64_t>(d1 * d2) \
        }; \
        return make_readonly(nb::ndarray<nb::numpy, double, nb::ndim<3>>( \
            out.ptr, 3, shape, nb::handle(), strides \
        )); \
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
        check_dimensions();

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

    // Allocation status
    bool get_allocated() const { return allocated; }

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
    DEFINE_FLAG_PROPERTY(old_intensity_correction, ds.flag.old_intensity_correction)
    DEFINE_FLAG_PROPERTY(spher, ds.flag.spher)
    DEFINE_INT_PROPERTY(brdf_type, ds.flag.brdf_type)

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

    // Accepts any contiguous array; nanobind converts to Fortran order if needed
    void set_pmom(ArrayD2F arr) {
        check_allocated();
        int d1 = ds.nmom_nstr + 1;
        int d2 = ds.nlyr;
        if (arr.shape(0) != d1 || arr.shape(1) != d2) {
            throw std::runtime_error(
                "pmom array shape mismatch. Expected ("
                + std::to_string(d1) + ", "
                + std::to_string(d2) + ")"
            );
        }
        // Direct copy — nanobind ensures arr is Fortran-contiguous
        std::copy_n(arr.data(), d1 * d2, ds.pmom);
    }

    // Return a view on cdisort's buffer with Fortran strides (no copy)
    auto get_pmom() {
        check_allocated();
        size_t shape[2] = {
            static_cast<size_t>(ds.nmom_nstr + 1),
            static_cast<size_t>(ds.nlyr)
        };
        // Fortran strides in elements (nanobind converts to bytes internally)
        int64_t strides[2] = {
            1,                              // dim 0: 1 element
            static_cast<int64_t>(ds.nmom_nstr + 1)  // dim 1: nmom_nstr+1 elements
        };
        // Return a non-owning view; caller must keep DisortState alive
        return nb::ndarray<nb::numpy, double, nb::ndim<2>>(
            ds.pmom, 2, shape, nb::handle(), strides
        );
    }

    DEFINE_ARRAY_PROPERTY(umu, ds.umu, ds.numu)
    DEFINE_ARRAY_PROPERTY(phi, ds.phi, ds.nphi)
    DEFINE_ARRAY_PROPERTY(utau, ds.utau, ds.ntau)
    DEFINE_ARRAY_PROPERTY(temper, ds.temper, ds.nlyr + 1)

    // Output accessors (read-only) - create copies owned by Python
    DEFINE_OUTPUT_RADIANT(rfldir, rfldir)
    DEFINE_OUTPUT_RADIANT(rfldn, rfldn)
    DEFINE_OUTPUT_RADIANT(flup, flup)
    DEFINE_OUTPUT_RADIANT(dfdt, dfdt)
    DEFINE_OUTPUT_RADIANT(uavg, uavg)
    DEFINE_OUTPUT_RADIANT(uavgdn, uavgdn)
    DEFINE_OUTPUT_RADIANT(uavgup, uavgup)
    DEFINE_OUTPUT_RADIANT(uavgso, uavgso)

    // Intensity and special boundary condition outputs
    DEFINE_OUTPUT_ARRAY_3D(uu, uu, ds.numu, ds.ntau, ds.nphi,
        "Intensity output not available. Ensure usrang=True, onlyfl=False, and run solve() first.")
    DEFINE_OUTPUT_ARRAY_2D(u0u, u0u, ds.numu, ds.ntau,
        "U0U output not available. Ensure usrang=True, onlyfl=False, and run solve() first.")
    DEFINE_OUTPUT_ARRAY_3D(uum, uum, ds.numu, ds.ntau, ds.nphi,
        "UUM output not available. Ensure output_uum=True and run solve() first.")
    DEFINE_OUTPUT_ARRAY_1D(albmed, albmed, ds.numu,
        "Albedo not available (only for ibcnd=SPECIAL_BC).")
    DEFINE_OUTPUT_ARRAY_1D(trnmed, trnmed, ds.numu,
        "Transmissivity not available (only for ibcnd=SPECIAL_BC).")

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
            throw std::runtime_error("Memory not allocated. Call allocate() first.");
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

    void check_dimensions() const {
        if (ds.nlyr <= 0) {
            throw std::runtime_error("nlyr must be positive, got " + std::to_string(ds.nlyr));
        }
        if (ds.nstr <= 0 || ds.nstr % 2 != 0) {
            throw std::runtime_error("nstr must be positive and even, got " + std::to_string(ds.nstr));
        }
    }
};

/*
 * BatchSolver class - parallel batch DISORT solver
 *
 * Holds shared (spectrally-constant) geometry/configuration as a prototype
 * disort_state, then creates nbatch copies with spectrally-varying optical
 * properties and solves them in parallel using a thread pool.
 *
 * Thread safety contract: cdisort has static variables that are not thread-safe.
 * We mitigate this by:
 * 1. Running a warm-up c_disort() call on the main thread to initialize all
 *    static guards before any parallel work.
 * 2. Requiring quiet=True (suppresses warning counter writes) and lamber=True
 *    (skips BRDF cached parameter writes).
 */
class BatchSolver {
public:
    // Prototype state — holds shared config (dimensions, flags, geometry)
    disort_state proto;
    int nbatch_;
    int nthreads_;
    bool allocated_;
    bool solved_;

    // Batch storage
    std::vector<disort_state> states_;
    std::vector<disort_output> outputs_;

    // Shared array buffers (set before allocate, copied to each state)
    std::vector<double> shared_umu_;
    std::vector<double> shared_phi_;
    std::vector<double> shared_utau_;
    std::vector<double> shared_temper_;

    // Thread pool
    std::unique_ptr<ThreadPool> pool_;

    BatchSolver(int nthreads = 0) : nbatch_(0), nthreads_(nthreads), allocated_(false), solved_(false) {
        memset(&proto, 0, sizeof(disort_state));

        // Register error callback
        static bool callback_registered = false;
        if (!callback_registered) {
            c_set_error_callback(cdisort_error_handler);
            callback_registered = true;
        }

        // Default nthreads to hardware concurrency
        if (nthreads_ <= 0) {
            nthreads_ = static_cast<int>(std::thread::hardware_concurrency());
            if (nthreads_ <= 0) nthreads_ = 4;
        }
    }

    ~BatchSolver() {
        free_batch();
    }

    // Non-copyable
    BatchSolver(const BatchSolver&) = delete;
    BatchSolver& operator=(const BatchSolver&) = delete;

    // --- Shared dimension properties ---
    DEFINE_INT_PROPERTY(nstr, proto.nstr)
    DEFINE_INT_PROPERTY(nlyr, proto.nlyr)
    DEFINE_INT_PROPERTY(nmom, proto.nmom)
    DEFINE_INT_PROPERTY(ntau, proto.ntau)
    DEFINE_INT_PROPERTY(numu, proto.numu)
    DEFINE_INT_PROPERTY(nphi, proto.nphi)

    // --- Shared flag properties ---
    DEFINE_FLAG_PROPERTY(usrtau, proto.flag.usrtau)
    DEFINE_FLAG_PROPERTY(usrang, proto.flag.usrang)
    DEFINE_FLAG_PROPERTY(lamber, proto.flag.lamber)
    DEFINE_FLAG_PROPERTY(planck, proto.flag.planck)
    DEFINE_FLAG_PROPERTY(onlyfl, proto.flag.onlyfl)
    DEFINE_FLAG_PROPERTY(quiet, proto.flag.quiet)
    DEFINE_FLAG_PROPERTY(intensity_correction, proto.flag.intensity_correction)
    DEFINE_FLAG_PROPERTY(old_intensity_correction, proto.flag.old_intensity_correction)
    DEFINE_FLAG_PROPERTY(spher, proto.flag.spher)

    // --- Shared scalar boundary conditions ---
    DEFINE_SCALAR_PROPERTY(umu0, proto.bc.umu0)
    DEFINE_SCALAR_PROPERTY(phi0, proto.bc.phi0)
    DEFINE_SCALAR_PROPERTY(fisot, proto.bc.fisot)
    DEFINE_SCALAR_PROPERTY(fluor, proto.bc.fluor)
    DEFINE_SCALAR_PROPERTY(btemp, proto.bc.btemp)
    DEFINE_SCALAR_PROPERTY(ttemp, proto.bc.ttemp)
    DEFINE_SCALAR_PROPERTY(temis, proto.bc.temis)

    // --- Shared scalar parameters ---
    DEFINE_SCALAR_PROPERTY(accur, proto.accur)
    DEFINE_SCALAR_PROPERTY(wvnmlo, proto.wvnmlo)
    DEFINE_SCALAR_PROPERTY(wvnmhi, proto.wvnmhi)

    // --- Shared array setters (stored locally, copied during allocate) ---
    void set_umu(ArrayD1 arr) {
        if (static_cast<int>(arr.shape(0)) != proto.numu)
            throw std::runtime_error("umu size mismatch. Expected " + std::to_string(proto.numu));
        shared_umu_.assign(arr.data(), arr.data() + arr.shape(0));
    }

    void set_phi(ArrayD1 arr) {
        if (static_cast<int>(arr.shape(0)) != proto.nphi)
            throw std::runtime_error("phi size mismatch. Expected " + std::to_string(proto.nphi));
        shared_phi_.assign(arr.data(), arr.data() + arr.shape(0));
    }

    void set_utau(ArrayD1 arr) {
        if (static_cast<int>(arr.shape(0)) != proto.ntau)
            throw std::runtime_error("utau size mismatch. Expected " + std::to_string(proto.ntau));
        shared_utau_.assign(arr.data(), arr.data() + arr.shape(0));
    }

    void set_temper(ArrayD1 arr) {
        if (static_cast<int>(arr.shape(0)) != proto.nlyr + 1)
            throw std::runtime_error("temper size mismatch. Expected " + std::to_string(proto.nlyr + 1));
        shared_temper_.assign(arr.data(), arr.data() + arr.shape(0));
    }

    // --- Allocation ---
    int get_nbatch() const { return nbatch_; }
    int get_nthreads() const { return nthreads_; }
    bool get_allocated() const { return allocated_; }
    bool get_solved() const { return solved_; }

    void allocate(int nbatch) {
        // Validate dimensions
        if (proto.nlyr <= 0)
            throw std::runtime_error("nlyr must be positive, got " + std::to_string(proto.nlyr));
        if (proto.nstr <= 0 || proto.nstr % 2 != 0)
            throw std::runtime_error("nstr must be positive and even, got " + std::to_string(proto.nstr));
        if (nbatch <= 0)
            throw std::runtime_error("nbatch must be positive, got " + std::to_string(nbatch));

        // Enforce thread safety contract
        if (proto.flag.quiet != TRUE)
            throw std::runtime_error("BatchSolver requires quiet=True for thread safety.");
        if (proto.flag.lamber != TRUE)
            throw std::runtime_error("BatchSolver requires lamber=True for thread safety.");

        // Warm up cdisort statics on the main thread
        ensure_warmup();

        // Free previous batch if any
        free_batch();

        nbatch_ = nbatch;
        solved_ = false;

        // Compute nmom_nstr (same logic as c_disort_state_alloc)
        proto.nmom_nstr = (proto.nmom > proto.nstr) ? proto.nmom : proto.nstr;

        // Create nbatch copies of prototype
        states_.resize(nbatch);
        outputs_.resize(nbatch);

        for (int i = 0; i < nbatch; ++i) {
            // Copy all scalar/flag fields from prototype
            memcpy(&states_[i], &proto, sizeof(disort_state));

            // Zero out pointers — alloc will set them
            states_[i].dtauc = nullptr;
            states_[i].ssalb = nullptr;
            states_[i].pmom = nullptr;
            states_[i].temper = nullptr;
            states_[i].utau = nullptr;
            states_[i].umu = nullptr;
            states_[i].phi = nullptr;
            states_[i].zd = nullptr;
            states_[i].mu_phase = nullptr;
            states_[i].phase = nullptr;
            states_[i].gensrc = nullptr;
            states_[i].gensrcu = nullptr;

            memset(&outputs_[i], 0, sizeof(disort_output));

            c_disort_state_alloc(&states_[i]);
            c_disort_out_alloc(&states_[i], &outputs_[i]);

            // Copy shared arrays into each state
            if (!shared_umu_.empty())
                std::copy_n(shared_umu_.data(), shared_umu_.size(), states_[i].umu);
            if (!shared_phi_.empty())
                std::copy_n(shared_phi_.data(), shared_phi_.size(), states_[i].phi);
            if (!shared_utau_.empty())
                std::copy_n(shared_utau_.data(), shared_utau_.size(), states_[i].utau);
            if (!shared_temper_.empty())
                std::copy_n(shared_temper_.data(), shared_temper_.size(), states_[i].temper);
        }

        // Create thread pool
        pool_ = std::make_unique<ThreadPool>(
            static_cast<size_t>(std::min(nthreads_, nbatch))
        );

        allocated_ = true;
    }

    // --- Batched input setters ---

    void set_dtauc(ArrayD2 arr) {
        check_allocated();
        check_batch_shape(arr, nbatch_, proto.nlyr, "dtauc");
        for (int i = 0; i < nbatch_; ++i)
            std::copy_n(arr.data() + i * proto.nlyr, proto.nlyr, states_[i].dtauc);
    }

    void set_ssalb(ArrayD2 arr) {
        check_allocated();
        check_batch_shape(arr, nbatch_, proto.nlyr, "ssalb");
        for (int i = 0; i < nbatch_; ++i)
            std::copy_n(arr.data() + i * proto.nlyr, proto.nlyr, states_[i].ssalb);
    }

    // pmom: (nmom_nstr+1, nlyr, nbatch)
    // Accepts any contiguous array; nanobind converts to Fortran order if needed
    void set_pmom(ArrayD3F arr) {
        check_allocated();
        int d1 = proto.nmom_nstr + 1;
        int d2 = proto.nlyr;
        if (static_cast<int>(arr.shape(0)) != d1 ||
            static_cast<int>(arr.shape(1)) != d2 ||
            static_cast<int>(arr.shape(2)) != nbatch_) {
            throw std::runtime_error(
                "pmom shape mismatch. Expected (" + std::to_string(d1)
                + ", " + std::to_string(d2)
                + ", " + std::to_string(nbatch_) + ") Fortran-contiguous, got ("
                + std::to_string(arr.shape(0)) + ", "
                + std::to_string(arr.shape(1)) + ", "
                + std::to_string(arr.shape(2)) + ")"
            );
        }
        int slice_size = d1 * d2;
        for (int i = 0; i < nbatch_; ++i)
            std::copy_n(arr.data() + i * slice_size, slice_size, states_[i].pmom);
    }

    void set_fbeam(ArrayD1 arr) {
        check_allocated();
        check_batch_1d(arr, nbatch_, "fbeam");
        for (int i = 0; i < nbatch_; ++i)
            states_[i].bc.fbeam = arr.data()[i];
    }

    void set_albedo(ArrayD1 arr) {
        check_allocated();
        check_batch_1d(arr, nbatch_, "albedo");
        for (int i = 0; i < nbatch_; ++i)
            states_[i].bc.albedo = arr.data()[i];
    }

    // Optional per-batch utau override: (nbatch, ntau)
    void set_utau_batched(ArrayD2 arr) {
        check_allocated();
        check_batch_shape(arr, nbatch_, proto.ntau, "utau");
        for (int i = 0; i < nbatch_; ++i)
            std::copy_n(arr.data() + i * proto.ntau, proto.ntau, states_[i].utau);
    }

    // --- Solve ---

    void solve() {
        check_allocated();

        std::vector<std::future<void>> futures;
        futures.reserve(nbatch_);

        {
            nb::gil_scoped_release release;

            for (int i = 0; i < nbatch_; ++i) {
                auto* ds_ptr = &states_[i];
                auto* out_ptr = &outputs_[i];
                futures.push_back(pool_->submit([ds_ptr, out_ptr]() {
                    c_disort(ds_ptr, out_ptr);
                }));
            }

            // Wait for all and propagate first exception
            std::exception_ptr first_exception;
            for (auto& f : futures) {
                try {
                    f.get();
                } catch (...) {
                    if (!first_exception)
                        first_exception = std::current_exception();
                }
            }
            if (first_exception)
                std::rethrow_exception(first_exception);
        }

        solved_ = true;
    }

    // --- Batched output accessors ---

    // Radiant quantity output: (nbatch, ntau)
    nb::object get_rfldir() { return get_batched_radiant(&disort_radiant::rfldir, "rfldir"); }
    nb::object get_rfldn()  { return get_batched_radiant(&disort_radiant::rfldn,  "rfldn");  }
    nb::object get_flup()   { return get_batched_radiant(&disort_radiant::flup,   "flup");   }
    nb::object get_dfdt()   { return get_batched_radiant(&disort_radiant::dfdt,   "dfdt");   }
    nb::object get_uavg()   { return get_batched_radiant(&disort_radiant::uavg,   "uavg");   }
    nb::object get_uavgdn() { return get_batched_radiant(&disort_radiant::uavgdn, "uavgdn"); }
    nb::object get_uavgup() { return get_batched_radiant(&disort_radiant::uavgup, "uavgup"); }
    nb::object get_uavgso() { return get_batched_radiant(&disort_radiant::uavgso, "uavgso"); }

    // Intensity output: (nbatch, numu, ntau, nphi)
    nb::object get_uu() {
        check_solved();
        int numu = proto.numu;
        int ntau = proto.ntau;
        int nphi = proto.nphi;
        int slice_size = numu * ntau * nphi;

        double* data = new double[nbatch_ * slice_size];
        for (int b = 0; b < nbatch_; ++b) {
            if (!outputs_[b].uu)
                throw std::runtime_error("Intensity output not available for batch " + std::to_string(b));
            // cdisort stores uu in Fortran order (numu, ntau, nphi)
            // We copy and reshape to (nbatch, numu, ntau, nphi) with Fortran inner strides
            std::copy_n(outputs_[b].uu, slice_size, data + b * slice_size);
        }

        size_t shape[4] = {
            static_cast<size_t>(nbatch_),
            static_cast<size_t>(numu),
            static_cast<size_t>(ntau),
            static_cast<size_t>(nphi)
        };
        // Strides: batch is outermost (C), inner dims are Fortran
        int64_t strides[4] = {
            static_cast<int64_t>(slice_size),  // batch stride
            1,                                  // numu stride (Fortran innermost)
            static_cast<int64_t>(numu),         // ntau stride
            static_cast<int64_t>(numu * ntau)   // nphi stride
        };
        nb::capsule owner(data, [](void* p) noexcept {
            delete[] static_cast<double*>(p);
        });
        return make_readonly(nb::ndarray<nb::numpy, double, nb::ndim<4>>(
            data, 4, shape, owner, strides
        ));
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "<BatchSolver"
            << " nstr=" << proto.nstr
            << " nlyr=" << proto.nlyr
            << " nmom=" << proto.nmom
            << " nbatch=" << nbatch_
            << " nthreads=" << nthreads_
            << " allocated=" << (allocated_ ? "True" : "False")
            << " solved=" << (solved_ ? "True" : "False")
            << ">";
        return oss.str();
    }

private:
    void check_allocated() const {
        if (!allocated_)
            throw std::runtime_error("Not allocated. Call allocate() first.");
    }

    void check_solved() const {
        check_allocated();
        if (!solved_)
            throw std::runtime_error("Not solved. Call solve() first.");
    }

    void check_batch_shape(const ArrayD2& arr, int expected_b, int expected_n, const char* name) const {
        if (static_cast<int>(arr.shape(0)) != expected_b || static_cast<int>(arr.shape(1)) != expected_n) {
            throw std::runtime_error(
                std::string(name) + " shape mismatch. Expected ("
                + std::to_string(expected_b) + ", " + std::to_string(expected_n)
                + "), got (" + std::to_string(arr.shape(0)) + ", " + std::to_string(arr.shape(1)) + ")"
            );
        }
    }

    void check_batch_1d(const ArrayD1& arr, int expected_b, const char* name) const {
        if (static_cast<int>(arr.shape(0)) != expected_b) {
            throw std::runtime_error(
                std::string(name) + " size mismatch. Expected "
                + std::to_string(expected_b) + ", got " + std::to_string(arr.shape(0))
            );
        }
    }

    nb::object get_batched_radiant(double disort_radiant::*field, const char* name) {
        check_solved();
        int ntau = proto.ntau;
        double* data = new double[nbatch_ * ntau];
        for (int b = 0; b < nbatch_; ++b) {
            if (!outputs_[b].rad)
                throw std::runtime_error(std::string(name) + " output not available for batch " + std::to_string(b));
            for (int t = 0; t < ntau; ++t)
                data[b * ntau + t] = outputs_[b].rad[t].*field;
        }
        size_t shape[2] = {static_cast<size_t>(nbatch_), static_cast<size_t>(ntau)};
        nb::capsule owner(data, [](void* p) noexcept {
            delete[] static_cast<double*>(p);
        });
        return make_readonly(nb::ndarray<nb::numpy, double, nb::ndim<2>>(data, 2, shape, owner));
    }

    void free_batch() {
        // Destroy thread pool first so workers finish before we free states
        pool_.reset();

        for (size_t i = 0; i < states_.size(); ++i) {
            c_disort_out_free(&states_[i], &outputs_[i]);
            c_disort_state_free(&states_[i]);
        }
        states_.clear();
        outputs_.clear();
        allocated_ = false;
        solved_ = false;
        nbatch_ = 0;
    }

    // One-time warm-up to initialize cdisort static variables
    static void ensure_warmup() {
        static bool warmed_up = false;
        if (warmed_up) return;

        disort_state ds;
        disort_output out;
        memset(&ds, 0, sizeof(disort_state));
        memset(&out, 0, sizeof(disort_output));

        ds.nstr = 2;
        ds.nlyr = 1;
        ds.nmom = 2;
        ds.ntau = 1;
        ds.numu = 0;
        ds.nphi = 0;
        ds.flag.usrtau = TRUE;
        ds.flag.usrang = FALSE;
        ds.flag.lamber = TRUE;
        ds.flag.planck = FALSE;
        ds.flag.onlyfl = TRUE;
        ds.flag.quiet = TRUE;
        ds.flag.ibcnd = GENERAL_BC;
        ds.flag.spher = FALSE;
        ds.flag.intensity_correction = FALSE;
        ds.flag.old_intensity_correction = FALSE;
        ds.bc.fbeam = M_PI;
        ds.bc.umu0 = 1.0;
        ds.bc.phi0 = 0.0;
        ds.bc.fisot = 0.0;
        ds.bc.fluor = 0.0;
        ds.bc.albedo = 0.0;
        ds.accur = 0.0;

        c_disort_state_alloc(&ds);
        c_disort_out_alloc(&ds, &out);

        ds.dtauc[0] = 1.0;
        ds.ssalb[0] = 0.5;
        ds.utau[0] = 0.0;
        // Isotropic phase function
        int nmom_nstr = (ds.nmom > ds.nstr) ? ds.nmom : ds.nstr;
        for (int k = 0; k <= nmom_nstr; ++k)
            ds.pmom[k] = (k == 0) ? 1.0 : 0.0;

        c_disort(&ds, &out);

        c_disort_out_free(&ds, &out);
        c_disort_state_free(&ds);

        warmed_up = true;
    }
};

// Nanobind binding macros for BatchSolver
#define BIND_BATCH_PROPERTY_RW(name, doc) \
    .def_prop_rw(#name, &BatchSolver::get_##name, &BatchSolver::set_##name, doc)

#define BIND_BATCH_PROPERTY_RO(name, doc) \
    .def_prop_ro(#name, &BatchSolver::get_##name, nb::rv_policy::move, doc)

/*
 * Python module definition
 */
NB_MODULE(_core, m) {
    m.doc() = "nanodisort core bindings to cdisort";

    // DisortState class
    nb::class_<DisortState>(m, "DisortState", "DISORT solver state")
        .def(nb::init<>(), "Create a new DISORT state.")
        .def("__repr__", &DisortState::repr)
        .def("allocate", &DisortState::allocate,
             "Allocate memory for arrays based on configured dimensions.")
        .def("solve", &DisortState::solve,
             "Run the DISORT solver.")
        // Allocation status
        BIND_PROPERTY_RO(allocated, "Memory allocation status")
        // Dimensions
        BIND_PROPERTY_RW(nstr, "Number of streams (discretization of the azimuth angle dimension).")
        BIND_PROPERTY_RW(nlyr, "Number of computational layers (discretization of the vertical dimension).")
        BIND_PROPERTY_RW(nmom, "Number of phase function moments. Set to -1 to disable scattering.")
        BIND_PROPERTY_RW(ntau, "Number of user optical thicknesses.")
        BIND_PROPERTY_RW(numu, "Number of user polar angles.")
        BIND_PROPERTY_RW(nphi, "Number of user azimuth angles.")
        // Control flags
        BIND_PROPERTY_RW(usrtau, "Return radiant quantities at user-specified optical thicknesses.")
        BIND_PROPERTY_RW(usrang, "Return radiant quantities at user-specified polar angles.")
        BIND_PROPERTY_RW(lamber, "Isotropically reflecting bottom boundary.")
        BIND_PROPERTY_RW(planck, "Include thermal emission.")
        BIND_PROPERTY_RW(onlyfl, "Return only fluxes (not intensities).")
        BIND_PROPERTY_RW(quiet, "Suppress output messages.")
        BIND_PROPERTY_RW(intensity_correction, "Apply intensity correction.")
        BIND_PROPERTY_RW(old_intensity_correction, "Apply old-style intensity correction (may be required for some tests).")
        BIND_PROPERTY_RW(spher, "Pseudo-spherical geometry (vs plane-parallel).")
        BIND_PROPERTY_RW(brdf_type, "Type of BRDF (0=none, 4=Hapke). Set lamber=False to activate.")
        // Boundary conditions
        BIND_PROPERTY_RW(fbeam, "Intensity of incident parallel beam.")
        BIND_PROPERTY_RW(umu0, "Polar angle cosine of incident beam.")
        BIND_PROPERTY_RW(phi0, "Azimuth angle of incident beam (degrees).")
        BIND_PROPERTY_RW(fisot, "Intensity of top-boundary isotropic illumination.")
        BIND_PROPERTY_RW(fluor, "Intensity of bottom-boundary isotropic illumination.")
        BIND_PROPERTY_RW(albedo, "Albedo of bottom boundary.")
        BIND_PROPERTY_RW(btemp, "Temperature [K] of bottom boundary.")
        BIND_PROPERTY_RW(ttemp, "Temperature [K] of top boundary.")
        BIND_PROPERTY_RW(temis, "Emissivity of top boundary.")
        // Other scalar parameters
        BIND_PROPERTY_RW(accur, "Convergence criterion for azimuthal series.")
        BIND_PROPERTY_RW(wvnmlo, "Wavenumber lower bound [cm⁻¹] for Planck function.")
        BIND_PROPERTY_RW(wvnmhi, "Wavenumber upper bound [cm⁻¹] for Planck function.")
        // Optical property arrays
        BIND_PROPERTY_RW(dtauc, "Optical thicknesses of computational layers [nlyr].")
        BIND_PROPERTY_RW(ssalb, "Single-scattering albedo of computational layers [nlyr].")
        BIND_PROPERTY_RW(pmom, "Phase function moments [nmom_nstr+1, nlyr].")
        // Other input arrays
        BIND_PROPERTY_RW(umu, "Polar angle cosines [numu].")
        BIND_PROPERTY_RW(phi, "Azimuthal angles [degrees] [nphi].")
        BIND_PROPERTY_RW(utau, "User optical thicknesses [ntau].")
        BIND_PROPERTY_RW(temper, "Temperatures [K] at levels [nlyr+1].")
        // Output arrays (read-only)
        BIND_PROPERTY_RO(rfldir, "Direct beam flux (without delta-M scaling) [ntau].")
        BIND_PROPERTY_RO(rfldn, "Diffuse downward flux (without delta-M scaling) [ntau].")
        BIND_PROPERTY_RO(flup, "Diffuse upward flux [ntau].")
        BIND_PROPERTY_RO(dfdt, "Flux divergence d(net flux)/d(optical thickness) [ntau].")
        BIND_PROPERTY_RO(uavg, "Mean intensity including direct beam [ntau].")
        BIND_PROPERTY_RO(uavgdn, "Mean diffuse downward intensity [ntau].")
        BIND_PROPERTY_RO(uavgup, "Mean diffuse upward intensity [ntau].")
        BIND_PROPERTY_RO(uavgso, "Mean direct solar intensity [ntau].")
        // Intensity outputs (require usrang=True, onlyfl=False)
        BIND_PROPERTY_RO(uu, "Intensity at user angles [numu, ntau, nphi].")
        BIND_PROPERTY_RO(u0u, "Azimuthally averaged intensity [numu, ntau].")
        BIND_PROPERTY_RO(uum, "Intensity (corrected) at user angles [numu, ntau, nphi].")
        // Special boundary condition outputs (require ibcnd=SPECIAL_BC)
        BIND_PROPERTY_RO(albmed, "Albedo of medium [numu].")
        BIND_PROPERTY_RO(trnmed, "Transmissivity of medium [numu].");

    // BatchSolver class
    nb::class_<BatchSolver>(m, "BatchSolver",
        "Parallel batch DISORT solver.\n\n"
        "Solves multiple spectral points in parallel using a thread pool.\n"
        "Shared geometry and flags are set once; spectrally-varying optical\n"
        "properties are provided as arrays with a leading batch dimension.\n\n"
        "Thread safety requires quiet=True and lamber=True.")
        .def(nb::init<int>(), "nthreads"_a = 0,
             "Create a new BatchSolver.\n\n"
             "Parameters\n----------\n"
             "nthreads : int, optional\n"
             "    Number of worker threads. Defaults to hardware concurrency.")
        .def("__repr__", &BatchSolver::repr)
        .def("allocate", &BatchSolver::allocate, "nbatch"_a,
             "Allocate memory for nbatch parallel DISORT problems.\n\n"
             "Must be called after setting shared dimensions and flags.")
        .def("solve", &BatchSolver::solve,
             "Solve all batch problems in parallel.")
        // Status
        BIND_BATCH_PROPERTY_RO(nbatch, "Number of batch items.")
        BIND_BATCH_PROPERTY_RO(nthreads, "Number of worker threads.")
        BIND_BATCH_PROPERTY_RO(allocated, "Memory allocation status.")
        BIND_BATCH_PROPERTY_RO(solved, "Whether solve() has been called.")
        // Shared dimensions
        BIND_BATCH_PROPERTY_RW(nstr, "Number of streams.")
        BIND_BATCH_PROPERTY_RW(nlyr, "Number of computational layers.")
        BIND_BATCH_PROPERTY_RW(nmom, "Number of phase function moments.")
        BIND_BATCH_PROPERTY_RW(ntau, "Number of user optical thicknesses.")
        BIND_BATCH_PROPERTY_RW(numu, "Number of user polar angles.")
        BIND_BATCH_PROPERTY_RW(nphi, "Number of user azimuth angles.")
        // Shared flags
        BIND_BATCH_PROPERTY_RW(usrtau, "Return radiant quantities at user-specified optical thicknesses.")
        BIND_BATCH_PROPERTY_RW(usrang, "Return radiant quantities at user-specified polar angles.")
        BIND_BATCH_PROPERTY_RW(lamber, "Isotropically reflecting bottom boundary (must be True).")
        BIND_BATCH_PROPERTY_RW(planck, "Include thermal emission.")
        BIND_BATCH_PROPERTY_RW(onlyfl, "Return only fluxes (not intensities).")
        BIND_BATCH_PROPERTY_RW(quiet, "Suppress output messages (must be True).")
        BIND_BATCH_PROPERTY_RW(intensity_correction, "Apply intensity correction.")
        BIND_BATCH_PROPERTY_RW(old_intensity_correction, "Apply old-style intensity correction.")
        BIND_BATCH_PROPERTY_RW(spher, "Pseudo-spherical geometry.")
        // Shared scalar boundary conditions
        BIND_BATCH_PROPERTY_RW(umu0, "Polar angle cosine of incident beam.")
        BIND_BATCH_PROPERTY_RW(phi0, "Azimuth angle of incident beam (degrees).")
        BIND_BATCH_PROPERTY_RW(fisot, "Intensity of top-boundary isotropic illumination.")
        BIND_BATCH_PROPERTY_RW(fluor, "Intensity of bottom-boundary isotropic illumination.")
        BIND_BATCH_PROPERTY_RW(btemp, "Temperature [K] of bottom boundary.")
        BIND_BATCH_PROPERTY_RW(ttemp, "Temperature [K] of top boundary.")
        BIND_BATCH_PROPERTY_RW(temis, "Emissivity of top boundary.")
        // Shared scalar parameters
        BIND_BATCH_PROPERTY_RW(accur, "Convergence criterion for azimuthal series.")
        BIND_BATCH_PROPERTY_RW(wvnmlo, "Wavenumber lower bound [cm⁻¹].")
        BIND_BATCH_PROPERTY_RW(wvnmhi, "Wavenumber upper bound [cm⁻¹].")
        // Shared array setters
        .def("set_umu", &BatchSolver::set_umu, "umu"_a,
             "Set shared polar angle cosines [numu].")
        .def("set_phi", &BatchSolver::set_phi, "phi"_a,
             "Set shared azimuthal angles [nphi].")
        .def("set_utau", &BatchSolver::set_utau, "utau"_a,
             "Set shared user optical thicknesses [ntau].")
        .def("set_temper", &BatchSolver::set_temper, "temper"_a,
             "Set shared level temperatures [nlyr+1].")
        // Batched input setters
        .def("set_dtauc", &BatchSolver::set_dtauc, "dtauc"_a,
             "Set optical thicknesses [nbatch, nlyr].")
        .def("set_ssalb", &BatchSolver::set_ssalb, "ssalb"_a,
             "Set single-scattering albedos [nbatch, nlyr].")
        .def("set_pmom", &BatchSolver::set_pmom, "pmom"_a,
             "Set phase function moments [nmom_nstr+1, nlyr, nbatch] Fortran-contiguous.")
        .def("set_fbeam", &BatchSolver::set_fbeam, "fbeam"_a,
             "Set incident beam intensities [nbatch].")
        .def("set_albedo", &BatchSolver::set_albedo, "albedo"_a,
             "Set bottom boundary albedos [nbatch].")
        .def("set_utau_batched", &BatchSolver::set_utau_batched, "utau"_a,
             "Set per-batch user optical thicknesses [nbatch, ntau].")
        // Batched outputs (read-only)
        BIND_BATCH_PROPERTY_RO(rfldir, "Direct beam flux [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(rfldn, "Diffuse downward flux [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(flup, "Diffuse upward flux [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(dfdt, "Flux divergence [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(uavg, "Mean intensity [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(uavgdn, "Mean diffuse downward intensity [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(uavgup, "Mean diffuse upward intensity [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(uavgso, "Mean direct solar intensity [nbatch, ntau].")
        BIND_BATCH_PROPERTY_RO(uu, "Intensity [nbatch, numu, ntau, nphi].");
}

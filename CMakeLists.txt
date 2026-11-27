# SPDX-FileCopyrightText: 2025 Rayference
#
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.15...3.27)
project(nanodisort VERSION 0.1.0 LANGUAGES C CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
# SKBUILD_SABI_COMPONENT (Development.SABIModule for abi3 builds) is required by
# nanobind's split mode on Windows to link against python3.lib.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
             ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

# cdisort library sources
set(CDISORT_SOURCES
    ext/cdisort-2.1.3/cdisort.c
    ext/cdisort-2.1.3/locate.c
)

set(CDISORT_HEADERS
    ext/cdisort-2.1.3/cdisort.h
    ext/cdisort-2.1.3/locate.h
)

# Build static cdisort library
add_library(cdisort STATIC ${CDISORT_SOURCES} ${CDISORT_HEADERS})
target_include_directories(cdisort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ext/cdisort-2.1.3)
set_target_properties(cdisort PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(cdisort PRIVATE -O3)
endif()
if(NOT WIN32)
    target_link_libraries(cdisort PUBLIC m)  # Link math library
endif()

# Python extension module using nanobind, built in split mode: the nanobind
# runtime is resolved at import time from the nanobind_backend module, so a
# single abi3 wheel per platform covers all supported Python versions.
nanobind_add_module(_core
    src/nanodisort/_core.cpp
    src/nanodisort/thread_pool.cpp
    BACKEND_MODULE nanobind_backend
)

# Link cdisort library to Python extension
target_link_libraries(_core PRIVATE cdisort Threads::Threads)

# The cdisort error callback throws a C++ exception through C frames. MSVC's
# default /EHsc assumes extern "C" functions never throw and may drop the
# handlers around c_disort() calls; /EHs keeps them.
if(MSVC)
    target_compile_options(_core PRIVATE /EHs)
endif()
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/cdisort-2.1.3
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nanodisort
)

# Install the extension module
install(TARGETS _core LIBRARY DESTINATION nanodisort COMPONENT python)

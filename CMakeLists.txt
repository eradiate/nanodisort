cmake_minimum_required(VERSION 3.15...3.27)
project(nanodisort VERSION 0.1.0 LANGUAGES C CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

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
target_compile_options(cdisort PRIVATE
    -O3
    -fPIC  # Position independent code for shared library
)
target_link_libraries(cdisort PUBLIC m)  # Link math library

# Python extension module using nanobind
nanobind_add_module(_core
    src/nanodisort/_core.cpp
    NB_STATIC  # Use static linking with nanobind
)

# Link cdisort library to Python extension
target_link_libraries(_core PRIVATE cdisort)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ext/cdisort-2.1.3)

# Install the extension module
install(TARGETS _core LIBRARY DESTINATION nanodisort)

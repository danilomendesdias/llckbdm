find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS $ENV{CONDA_PREFIX}/include)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})

set(COR_LIB "libmkl_core.dylib")
set(RT_LIB "libmkl_rt.dylib")
set(SEQ "libmkl_sequential.dylib")
set(ILP "libmkl_intel_ilp64.dylib")

find_library(MKL_CORE_LIBRARY
        NAMES ${COR_LIB}
        PATHS $ENV{CONDA_PREFIX}/lib
        NO_DEFAULT_PATH)

find_library(MKL_RT
        NAMES ${RT_LIB}
        PATHS $ENV{CONDA_PREFIX}/lib
        NO_DEFAULT_PATH)

find_library(MKL_SEQ
        NAMES ${SEQ}
        PATHS $ENV{CONDA_PREFIX}/lib
        NO_DEFAULT_PATH)

find_library(MKL_ILP
        NAMES ${ILP}
        PATHS $ENV{CONDA_PREFIX}/lib
        NO_DEFAULT_PATH)

set(MKL_LIBRARIES ${MKL_CORE_LIBRARY} ${MKL_SEQ}  ${MKL_RT}  ${MKL_ILP})


message("MKL_INCLUDE_DIRS: ${MKL_INCLUDE_DIRS}")
message("MKL_CORE_LIBRARY: ${MKL_CORE_LIBRARY}")
message("MKL_RT: ${MKL_RT}")

include_directories(${MKL_INCLUDE_DIRS})


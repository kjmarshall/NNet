## ---------------------------------------------------- ##
## Include Source and Build Directories in Include Path ##
## ---------------------------------------------------- ##
set(CMAKE_INCLUDE_CURRENT_DIR ON)

## ------------------------------ ##
## Prefer Includes in Source Tree ##
## ------------------------------ ##
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

## ------------------ ##
## Use Colored Output ##
## ------------------ ##
set(CMAKE_COLOR_MAKEFILE ON)

## ----------------------- ##
## Setup Build Environment ##
## ----------------------- ##
if(CMAKE_BUILD_TYPE STREQUAL "")
  # CMake defaults to leaving CMAKE_BUILD_TYPE empty. This screws up differentiation between debug and release builds.
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

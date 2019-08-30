## ------------ ##
## Project Name ##
## ------------ ##
project(CMAKE-PROJECT-SKELETON)
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
message(STATUS "PROCESSING ${PROJECT_NAME}")

## --------------- ##
## Project Options ##
## --------------- ##
option( ${PROJECT_NAME}_OPTION1                 "Option1"  ON )
option( ${PROJECT_NAME}_OPTION2                 "Option2"  ON )

## ------- ##
## VERBOSE ##
## ------- ##
# set( CMAKE_VERBOSE_MAKEFILE on )

## -------------- ##
## Version Number ##
## -------------- ##
set(${PROJECT_NAME}_VERSION_MAJOR 1)
set(${PROJECT_NAME}_VERSION_MINOR 0)
set(${PROJECT_NAME}_VERSION_PATCH 0)
set(${PROJECT_NAME}_VERSION_TWEAK 0)
set(${PROJECT_NAME}_VERSION
  "${PROJECT_NAME}_VERSION_MAJOR.${PROJECT_NAME}_VERSION_MINOR.${PROJECT_NAME}_VERSION_PATCH.${PROJECT_NAME}_VERSION_TWEAK")

## --------------- ##
## Set Module Path ##
## --------------- ##
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake-modules)

## ---------------------- ##
## Include Custom Modules ##
## --------------------- ##
include(compiler-flags)
include(custom-cmake-settings)
include(helper-functions)

## -------------- ##
## Chosen Options ##
## -------------- ##
set( INCLUDES "" )
set( LIBS "" )
set( EXTRA_LIBS "" )
set( DEFINITIONS "" )

## ---------------- ##
## Check for OpenMP ##
## ---------------- ##
# find_package(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

## ----- ##
## Eigen ##
## ----- ##
set( EIGEN_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3")
set( EIGEN_INCLUDE_DIRS ${EIGEN_DIR} )
list( APPEND INCLUDES "${EIGEN_INCLUDE_DIRS}" )

## ---- ##
## CUDA ##
## ---- ##
# find_package( CUDA REQUIRED )
# if ( CUDA_FOUND )
#   list( APPEND INCLUDES ${CUDA_INCLUDE_DIRS} )
#   list( APPEND LIBS ${CUDA_LIBRARIES} )
#   # nvcc flags
#   list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
#   # set( CUDA_PROPAGATE_HOST_FLAGS OFF )
# endif( )

## ----------- ##
## Google Test ##
## ----------- ##
option(PACKAGE_TESTS "Build the tests" ON)
if(PACKAGE_TESTS)
  enable_testing()
endif()
set( GOOGLE_TEST_DIR "${PROJECT_SOURCE_DIR}/external/googletest" )
add_subdirectory( "${GOOGLE_TEST_DIR}" EXCLUDE_FROM_ALL )
mark_as_advanced(
  BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
  gmock_build_tests gtest_build_samples gtest_build_tests
  gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )
set_target_properties(gtest PROPERTIES FOLDER external)
set_target_properties(gtest_main PROPERTIES FOLDER external)
set_target_properties(gmock PROPERTIES FOLDER external)
set_target_properties(gmock_main PROPERTIES FOLDER external)
list( APPEND INCLUDES "${GOOGLE_TEST_DIR}/googletest/include" )

## ------------ ##
## Add projects ##
## ------------ ##
include_directories( "${INCLUDES}" )
include_directories( "${PROJECT_SOURCE_DIR}/source" )
include_directories( "${PROJECT_SOURCE_DIR}/source/nnet" )
add_subdirectory( source )
add_subdirectory( tests )

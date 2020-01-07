## --------------------- ##
## CMake minimum version ##
## --------------------- ##
cmake_minimum_required(VERSION 3.1...3.15 FATAL_ERROR)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

## ---------------------------- ##
## Require out of source builds ##
## ---------------------------- ##
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

## ------------ ##
## Project Name ##
## ------------ ##
project(NNET LANGUAGES CXX)
message(STATUS "PROCESSING ${PROJECT_NAME}")

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
## Project Options ##
## --------------- ##
option(GIT_SUBMODULE "Check submodules during build" ON)

## ----------------- ##
## Update Submodules ##
## ----------------- ##
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

## ------- ##
## VERBOSE ##
## ------- ##
# set( CMAKE_VERBOSE_MAKEFILE on )

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

## -------------------- ##
## GCC compiler options ##
## -------------------- ##
# if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ftemplate-depth=1000")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

#   # To remove unused functions warnings.
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
# endif()

## -------------- ##
## Chosen Options ##
## -------------- ##
set( INCLUDES "" )
set( LIBS "" )
set( EXTRA_LIBS "" )
set( DEFINITIONS "" )

## --------------- ##
## Check for Boost ##
## --------------- ##
find_package( Boost REQUIRED COMPONENTS serialization )
if ( Boost_FOUND )
  include_directories( "${Boost_INCLUDE_DIRS}" )
  link_directories( "${Boost_LIBRARY_DIRS}" )
  list( APPEND LIBS "${Boost_LIBRARIES}" )
elseif()
  message("Boost not found.")
endif()

## ----- ##
## Eigen ##
## ----- ##
set( EIGEN_DIR "${CMAKE_SOURCE_DIR}/external/eigen-3.3")
set( EIGEN_INCLUDE_DIRS ${EIGEN_DIR} )
list( APPEND INCLUDES "${EIGEN_INCLUDE_DIRS}" )

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
#list( APPEND INCLUDES "${GOOGLE_TEST_DIR}/googletest/include" )

## ------------ ##
## Add projects ##
## ------------ ##
include_directories( "${INCLUDES}" )
include_directories( "${PROJECT_SOURCE_DIR}/source" )
include_directories( "${PROJECT_SOURCE_DIR}/source/nnet" )
add_subdirectory( source )
add_subdirectory( tests )

## ---------------------- ##
## Project: various tests ##
## ---------------------- ##
project( google-tests )
message(STATUS "PROCESSING ${PROJECT_NAME}")

## -------- ##
## Includes ##
## -------- ##
sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)

## ---------- ##
## Executable ##
## ---------- ##
message( STATUS "HEADER_FILES: ${HEADER_FILES}" )
message( STATUS "SOURCE_FILES: ${SOURCE_FILES}" )
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
sdk_setup_project_bin(${PROJECT_NAME})
add_dependencies(${PROJECT_NAME} nnet )
target_link_libraries(${PROJECT_NAME} gtest gmock gtest_main ${LIBS} ${EXTRA_LIBS})

# macro(package_add_test TESTNAME)
#     add_executable(${TESTNAME} ${ARGN})
#     target_link_libraries(${TESTNAME} gtest gmock gtest_main)
#     add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
#     set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
# endmacro()
# package_add_test( dumbtest test.cpp )

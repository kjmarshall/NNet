## ------------- ##
## Project: nnet ##
## ------------- ##
project(nnet)
message(STATUS "PROCESSING ${PROJECT_NAME}")

## -------- ##
## Includes ##
## -------- ##
sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)

## ----------------------- ##
## Interface Library Build ##
## ----------------------- ##
add_library(${PROJECT_NAME} INTERFACE)
target_compile_definitions(${PROJECT_NAME} INTERFACE ${DEFINITIONS})

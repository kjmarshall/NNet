## ---------------------------------- ##
## Functions to make our lives easier ##
## ---------------------------------- ##
# Function to list all header files in the current directory, recursing into sub-directories
# HEADER_FILES - To be filled with the found header files
function(sdk_list_header_files HEADER_FILES)
  file(GLOB_RECURSE HEADER_FILES_TMP "*.h" "*.hpp" "*.inl" "*.pch" "*.cuh")
  set(HEADER_FILES ${HEADER_FILES_TMP} PARENT_SCOPE)
endfunction()

# Function to list all source files in the current directory, recursing into sub-directories
# SOURCE_FILES - To be filled with the found source files
function(sdk_list_source_files SOURCE_FILES)
  file(GLOB_RECURSE SOURCE_FILES_TMP "*.c" "*.cpp" "*.cu")
  set(SOURCE_FILES ${SOURCE_FILES_TMP} PARENT_SCOPE)
endfunction()

# Function to setup some standard project items
# PROJECTNAME - The name of the project being setup
# TARGETDIR - The target directory for output files (relative to CMAKE_SOURCE_DIR)
function(sdk_setup_project_common PROJECTNAME TARGETDIR)
  # Set the Debug and Release names
  set_target_properties(
    ${PROJECTNAME}
    PROPERTIES
    DEBUG_OUTPUT_NAME ${PROJECTNAME}_d
    RELEASE_OUTPUT_NAME ${PROJECTNAME}
    )

  # Setup install to copy the built output to the target directory
  # (for compilers that don't have post build steps)
  install(
    TARGETS ${PROJECTNAME}
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/${TARGETDIR}"
    ARCHIVE DESTINATION "${CMAKE_SOURCE_DIR}/${TARGETDIR}"
    RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/${TARGETDIR}"
    )
endfunction()

# Function to setup some project items for an executable or DLL
# PROJECTNAME - The name of the project being setup
function(sdk_setup_project_bin PROJECTNAME)
  sdk_setup_project_common(${PROJECTNAME} bin)
endfunction()

# Function to setup some project items for static library
# PROJECTNAME - The name of the project being setup
function(sdk_setup_project_lib PROJECTNAME)
  sdk_setup_project_common(${PROJECTNAME} lib)
endfunction()

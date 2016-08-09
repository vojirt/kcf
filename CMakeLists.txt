cmake_minimum_required(VERSION 2.8)

project(kcf_tracker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

FIND_PACKAGE( OpenCV REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")

include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( ${CMAKE_BINARY_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/src/piotr_fhog ${CMAKE_SOURCE_DIR}/src/cn)

add_subdirectory(src)

add_executable(kcf_vot main_vot.cpp vot.hpp)
target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf)

# Try to find TraX header and library ...
FIND_FILE(TRAX_HEADER NAMES trax.h PATHS /home/vojirtom/qBT_dir/test/vot-toolkit-2016/native)
IF (EXISTS ${TRAX_HEADER})
    GET_FILENAME_COMPONENT(TRAX_DIR ${TRAX_HEADER} DIRECTORY)
    ADD_DEFINITIONS(-DTRAX) # Added to enable TraX protocol support
    LINK_DIRECTORIES(${TRAX_DIR}) # Add library directory
    INCLUDE_DIRECTORIES(${TRAX_DIR}) # Add includes directory
ENDIF()

IF (EXISTS ${TRAX_HEADER})
    add_executable(kcf_trax main_vot_trax.cpp vot_trax.h)
    target_link_libraries(kcf_trax ${OpenCV_LIBS} trax kcf)
ENDIF()



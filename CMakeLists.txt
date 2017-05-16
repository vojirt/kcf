cmake_minimum_required(VERSION 2.8)

project(kcf_tracker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

FIND_PACKAGE( OpenCV REQUIRED )
link_directories ( ${OpenCV_LIB_DIR} )
MESSAGE(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
ENDIF ()

include_directories ( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( ${CMAKE_BINARY_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/ ${CMAKE_SOURCE_DIR}/src/piotr_fhog ${CMAKE_SOURCE_DIR}/src/cn)

add_subdirectory(src)

add_executable(kcf_vot main_vot.cpp vot.hpp)
target_link_libraries(kcf_vot ${OpenCV_LIBS} kcf)

# Try to find TraX header and library ...
FIND_PACKAGE(trax REQUIRED COMPONENTS core opencv)
IF (trax_FOUND)
    MESSAGE(STATUS "Trax protocol found: building kcf_trax")
    INCLUDE_DIRECTORIES(${TRAX_INCLUDE_DIRS})
    LINK_DIRECTORIES(${TRAX_LIBRARY_DIRS})

    add_executable(kcf_trax main_trax.cpp)
    target_link_libraries(kcf_trax ${OpenCV_LIBS} kcf ${TRAX_LIBRARIES})
ENDIF()


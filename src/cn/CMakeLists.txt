cmake_minimum_required(VERSION 2.8)

set(CN_LIB_SRC cn_data.cpp cnfeat.hpp)

add_library(cndata STATIC ${CN_LIB_SRC})
target_link_libraries(cndata)
set_target_properties(cndata PROPERTIES VERSION 1.0.0 SOVERSION 1)
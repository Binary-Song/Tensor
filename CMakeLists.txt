cmake_minimum_required(VERSION 3.19)
project(Tensor)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(AddModule)
include(CTest)

if(BUILD_TESTING)
    set(Boost_USE_STATIC_LIBS ON) 
    set(Boost_USE_MULTITHREADED ON)
    find_package(Boost 1.74.0 REQUIRED COMPONENTS
        unit_test_framework
        regex
    )
endif() 
add_subdirectory(Tensor)
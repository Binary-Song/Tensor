cmake_minimum_required(VERSION 3.18)
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

macro(SetStandard STD) 
    message(c++${STD})
    set(CMAKE_CXX_STANDARD_REQUIRED ON ) # standard mandatory
    set(CMAKE_CXX_EXTENSIONS        OFF ) # no gnu c++
    set(CMAKE_CXX_STANDARD          ${STD}) # min standard 
endmacro()

SetStandard(17)

add_subdirectory(Tensor)
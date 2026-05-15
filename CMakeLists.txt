cmake_minimum_required(VERSION 3.14)
project(myln-frame VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release: max optimization, no GPU flags needed
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

include_directories(include)

add_executable(hello_myln examples/hello_myln.cpp)

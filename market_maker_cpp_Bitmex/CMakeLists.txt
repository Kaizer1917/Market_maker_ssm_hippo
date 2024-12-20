cmake_minimum_required(VERSION 3.15)
project(market_maker_cpp_bitmex VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTING "Build tests" ON)
option(USE_CUDA "Enable CUDA support" ON)
option(USE_TVM "Enable TVM support" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Find dependencies
find_package(Torch REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Protobuf REQUIRED)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
endif()

if(USE_TVM)
    find_package(TVM REQUIRED)
endif()

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${TVM_INCLUDE_DIRS}
)

# Library sources
file(GLOB_RECURSE SOURCES 
    "src/market_maker/core/*.cpp"
    "src/market_maker/model/*.cpp"
    "src/market_maker/strategy/*.cpp"
    "src/market_maker/utils/*.cpp"
    "src/market_maker/backtest/*.cpp"
)

# Create library
add_library(market_maker SHARED ${SOURCES})
target_link_libraries(market_maker PUBLIC
    ${TORCH_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    nlohmann_json::nlohmann_json
    protobuf::libprotobuf
)

if(USE_CUDA)
    target_link_libraries(market_maker PUBLIC ${CUDA_LIBRARIES})
endif()

if(USE_TVM)
    target_link_libraries(market_maker PUBLIC ${TVM_LIBRARIES})
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS market_maker
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
) 
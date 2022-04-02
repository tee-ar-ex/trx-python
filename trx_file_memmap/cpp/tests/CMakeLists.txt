cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx)
set (CMAKE_CXX_STANDARD 11)

set(PROJECT_BINARY_DIR ../../builds)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/tests)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_BUILD_TYPE Debug)

find_package(libzip REQUIRED)
find_package (Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

enable_testing()

include_directories(../src)
add_executable(test_mmap test_trx_mmap.cpp ../src/trx.h ../src/trx.tpp ../src/trx.cpp)


TARGET_LINK_LIBRARIES(test_mmap
	nlohmann_json::nlohmann_json
	Eigen3::Eigen
	libzip::zip
	GTest::gtest
	GTest::gtest_main
)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
include(GoogleTest)
gtest_discover_tests(test_mmap)

cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17")

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)
find_package(libzip CONFIG REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(jsoncpp CONFIG REQUIRED)

# include_directories(tests)
# link_directories(tests)
# add_subdirectory(tests)
add_library(trx src/trx.cpp src/trx.tpp src/trx.h)

TARGET_LINK_LIBRARIES(trx
	PRIVATE
	jsoncpp
	ZLIB::ZLIB
	libzip::zip
	Eigen3::Eigen
)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

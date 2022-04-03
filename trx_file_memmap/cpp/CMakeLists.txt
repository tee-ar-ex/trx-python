cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

find_package(libzip REQUIRED)
find_package (Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(trx src/trx.cpp src/trx.tpp src/trx.h)

TARGET_LINK_LIBRARIES(trx
	PRIVATE
	nlohmann_json::nlohmann_json
	libzip::zip
	Eigen3::Eigen
)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

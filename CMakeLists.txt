cmake_minimum_required(VERSION 3.27)

project(
  gnuradio4Algorithm
  VERSION 4.0.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(FetchContent)
find_package(PkgConfig QUIET)

option(ENABLE_EXAMPLES "Enable example builds" ON)
option(ENABLE_TESTING "Enable test builds" ON)
option(GR_USE_FETCHCONTENT_DEPS "Allow CMake to download selected third-party dependencies with FetchContent" OFF)
option(USE_CCACHE "Use ccache if available" ON)
set(GR_HTTP_ENABLED OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND USE_CCACHE)
  message(STATUS "ccache found and will be used")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
else()
  message(STATUS "ccache will not be used")
endif()

if(ENABLE_TESTING
   AND (UNIX
        OR APPLE
        OR WIN32))
  include(CTest)
  enable_testing()
endif()

find_package(gnuradio4 CONFIG REQUIRED)

if(NOT TARGET gnuradio-core AND TARGET gnuradio4::gnuradio-core)
  add_library(gnuradio-core ALIAS gnuradio4::gnuradio-core)
endif()
if(NOT TARGET gnuradio-options AND TARGET gnuradio4::gnuradio-options)
  add_library(gnuradio-options ALIAS gnuradio4::gnuradio-options)
endif()
if(NOT TARGET gnuradio-blocklib-core AND TARGET gnuradio4::gnuradio-blocklib-core)
  add_library(gnuradio-blocklib-core ALIAS gnuradio4::gnuradio-blocklib-core)
endif()
if(NOT TARGET gnuradio-meta AND TARGET gnuradio4::gnuradio-meta)
  add_library(gnuradio-meta ALIAS gnuradio4::gnuradio-meta)
endif()
if(NOT TARGET vir AND TARGET gnuradio4::vir)
  add_library(vir ALIAS gnuradio4::vir)
endif()
if(NOT TARGET magic_enum AND TARGET gnuradio4::magic_enum)
  add_library(magic_enum ALIAS gnuradio4::magic_enum)
endif()

find_package(httplib CONFIG QUIET)
if(NOT TARGET httplib::httplib)
  pkg_search_module(
    CPP_HTTPLIB
    QUIET
    IMPORTED_TARGET
    cpp-httplib
    httplib)
  if(CPP_HTTPLIB_FOUND)
    add_library(httplib::httplib INTERFACE IMPORTED)
    target_link_libraries(httplib::httplib INTERFACE PkgConfig::CPP_HTTPLIB)
  endif()
endif()
if(NOT TARGET httplib::httplib)
  message(FATAL_ERROR "cpp-httplib is required to build the algorithm tests.")
endif()

set(BOOST_UT_DISABLE_MODULE
    ON
    CACHE BOOL "Disable UT Module Support" FORCE)
add_compile_definitions(BOOST_UT_DISABLE_MODULE)
if(EMSCRIPTEN OR GR_USE_FETCHCONTENT_DEPS)
  FetchContent_Declare(
    ut
    GIT_REPOSITORY https://github.com/boost-ext/ut.git
    GIT_TAG 53e17f25119598c6458d30351b260193096ba67e
    EXCLUDE_FROM_ALL)
else()
  find_package(ut CONFIG QUIET)
  if(TARGET Boost::ut AND NOT TARGET ut)
    add_library(ut ALIAS Boost::ut)
  elseif(TARGET boost::ut AND NOT TARGET ut)
    add_library(ut ALIAS boost::ut)
  endif()
  if(NOT TARGET ut)
    find_path(GR_BOOST_UT_INCLUDE_DIR boost/ut.hpp)
    if(NOT GR_BOOST_UT_INCLUDE_DIR)
      message(FATAL_ERROR "Boost.UT was not found. Install libboost-ext-ut-dev, ut-devel, or a source-built Boost.UT.")
    endif()
    add_library(ut INTERFACE)
    target_include_directories(ut INTERFACE ${GR_BOOST_UT_INCLUDE_DIR})
  endif()
endif()
if(EMSCRIPTEN OR GR_USE_FETCHCONTENT_DEPS)
  FetchContent_MakeAvailable(ut)
endif()
if(NOT TARGET Boost::ut AND TARGET ut)
  add_library(Boost::ut ALIAS ut)
endif()

if(NOT TARGET exprtk)
  add_library(exprtk STATIC third_party/exprtk.cpp)
  set_target_properties(exprtk PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(
    exprtk
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endif()
if(NOT TARGET gnuradio4::exprtk)
  add_library(gnuradio4::exprtk ALIAS exprtk)
endif()

set(GR4_EXPORT_NAME gnuradio4AlgorithmTargets)
set(GR4_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/gnuradio4Algorithm")
set(GR4_PKG_VERSION "${PROJECT_VERSION}-git" CACHE STRING "Version string used in gnuradio4Algorithm.pc")
set(GR_TEST_HELPER_LIBRARIES "")

if(ENABLE_TESTING)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GnuRadioTestHelpers.cmake)
endif()

add_subdirectory(bench)
add_subdirectory(algorithm)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gnuradio4AlgorithmConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4AlgorithmConfig.cmake"
  INSTALL_DESTINATION "${GR4_CMAKE_INSTALL_DIR}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4AlgorithmConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

configure_file(cmake/gnuradio4Algorithm.pc.in "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4Algorithm.pc" @ONLY)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4AlgorithmConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4AlgorithmConfigVersion.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/gnuradio4Algorithm.pc"
  DESTINATION "${GR4_CMAKE_INSTALL_DIR}")

install(TARGETS exprtk EXPORT ${GR4_EXPORT_NAME} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES $<TARGET_FILE:exprtk> DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES third_party/exprtk.hpp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY third_party/exprtk DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT ${GR4_EXPORT_NAME} NAMESPACE gnuradio4:: DESTINATION "${GR4_CMAKE_INSTALL_DIR}")

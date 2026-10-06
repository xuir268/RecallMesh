cmake_minimum_required(VERSION 3.26)
project(assoc_mem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_subdirectory(engine)

option(ASSOC_BUILD_PYTHON "Build the Python extension" ON)
if(ASSOC_BUILD_PYTHON)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_engine STABLE_ABI NB_STATIC bindings/module.cpp)
target_link_libraries(_engine PRIVATE assoc_engine)

install(TARGETS _engine LIBRARY DESTINATION assoc_mem)

endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(assoc_smoke tests/engine_smoke.cpp)
  target_link_libraries(assoc_smoke PRIVATE assoc_engine)
  add_test(NAME assoc_smoke COMMAND assoc_smoke)
endif()

if(BUILD_TESTING)
  add_executable(csr_build_stress tests/csr_build_stress.cpp)
  target_link_libraries(csr_build_stress PRIVATE assoc_engine)
  add_test(NAME csr_build_stress COMMAND csr_build_stress)
endif()

option(ASSOC_BUILD_BENCHMARKS "Build standalone concurrency benchmark" OFF)
if(ASSOC_BUILD_BENCHMARKS)
  add_executable(concurrent_agents benchmarks/concurrent_agents.cpp)
  target_link_libraries(concurrent_agents PRIVATE assoc_engine)
endif()

if(BUILD_TESTING)
  add_executable(segment_recovery tests/segment_recovery.cpp)
  target_link_libraries(segment_recovery PRIVATE assoc_engine)
  add_test(NAME segment_recovery COMMAND segment_recovery)
endif()

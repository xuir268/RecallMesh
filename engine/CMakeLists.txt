# Pure C++ core. No Python anywhere, so it builds and profiles standalone --
# when chasing a contention bug you do not want the interpreter in the trace.
add_library(assoc_engine STATIC
  src/edge_store.cpp
  src/segment.cpp
  src/csr.cpp
)

target_include_directories(assoc_engine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(assoc_engine PUBLIC cxx_std_20)

# -fPIC: the static lib is linked into a shared Python extension.
set_target_properties(assoc_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
  target_compile_options(assoc_engine PRIVATE -Wall -Wextra)
endif()

find_package(Threads REQUIRED)
target_link_libraries(assoc_engine PUBLIC Threads::Threads)

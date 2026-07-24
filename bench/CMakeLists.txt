add_library(ut-benchmark INTERFACE benchmark.hpp)
if(NOT TARGET gnuradio4::ut-benchmark)
  add_library(gnuradio4::ut-benchmark ALIAS ut-benchmark)
endif()
target_include_directories(ut-benchmark INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

function(add_benchmark BM_NAME)
  add_executable(${BM_NAME} ${BM_NAME}.cpp)
  target_link_libraries(${BM_NAME} PRIVATE gnuradio-options ut gnuradio4::ut-benchmark)
endfunction()

add_benchmark(bm_example)

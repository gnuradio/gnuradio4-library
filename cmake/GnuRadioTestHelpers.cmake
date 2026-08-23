function(setup_test_no_asan TEST_NAME)
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE gnuradio-options
            gnuradio-core
            gnuradio-blocklib-core
            ut
            ${GR_TEST_HELPER_LIBRARIES})
  add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
endfunction()

function(setup_test TEST_NAME)
  setup_test_no_asan(${TEST_NAME})
endfunction()

function(add_ut_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
  if(GR_QA_OPTIMIZATION_LEVEL)
    # GCC's null-dereference analysis false-positives in libstdc++'s inlined string code below -O2; the warning battery
    # reads this property.
    set_target_properties(${TEST_NAME} PROPERTIES GR_QA_REDUCED_OPTIMIZATION ON)
    target_compile_options(${TEST_NAME} PRIVATE $<$<NOT:$<CONFIG:Debug>>:${GR_QA_OPTIMIZATION_LEVEL}>)
  endif()
  setup_test(${TEST_NAME})
  set_property(TEST ${TEST_NAME} PROPERTY ENVIRONMENT_MODIFICATION
                                          "GNURADIO4_PLUGIN_DIRECTORIES=set:${CMAKE_CURRENT_BINARY_DIR}/plugins")
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
endfunction()

# Pin a test above GR_QA_OPTIMIZATION_LEVEL and restore the diagnostics that only hold at the higher level: the warning
# battery reads GR_QA_REDUCED_OPTIMIZATION to decide whether GCC's null-dereference analysis is trustworthy for this
# target.
function(gr_pin_test_optimization TEST_NAME LEVEL)
  target_compile_options(${TEST_NAME} PRIVATE $<$<NOT:$<CONFIG:Debug>>:${LEVEL}>)
  set_target_properties(${TEST_NAME} PROPERTIES GR_QA_REDUCED_OPTIMIZATION OFF)
endfunction()

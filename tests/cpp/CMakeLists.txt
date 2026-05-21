# Each test file is its own executable. Keeps test failures localized
# and lets `ctest -R` filter cleanly. Tests link against the engine library
# and can additionally include private headers from src/.

function(add_engine_test test_name)
    add_executable(${test_name} ${test_name}.cpp test_main.cpp)
    target_link_libraries(${test_name} PRIVATE tensor_engine_objs)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_engine_test(test_allocator)
add_engine_test(test_storage)
add_engine_test(test_small_vec)
add_engine_test(test_tensor_impl)
add_engine_test(test_broadcast)
add_engine_test(test_ops_elementwise)
add_engine_test(test_ops_reduce)
add_engine_test(test_ops_matmul)
add_engine_test(test_ops_misc)
add_engine_test(test_ffi)

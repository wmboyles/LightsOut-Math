cmake_minimum_required(VERSION 3.20)
project(LightsOutNTL LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(NTL REQUIRED IMPORTED_TARGET ntl)
find_package(Threads REQUIRED)

add_executable(ntl_gf2x_gcd ntl_gf2x_gcd.cpp)
target_compile_features(ntl_gf2x_gcd PRIVATE cxx_std_11)
target_link_libraries(ntl_gf2x_gcd PRIVATE PkgConfig::NTL Threads::Threads)
if(WIN32)
    target_link_libraries(ntl_gf2x_gcd PRIVATE psapi)
endif()

set_target_properties(ntl_gf2x_gcd PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(ntl_gf2x_gcd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

if(WIN32)
    set(NTL_RUNTIME_DIRECTORIES "" CACHE STRING "Additional NTL runtime DLL directories")
    get_filename_component(_compiler_directory "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_runtime_directories ${NTL_RUNTIME_DIRECTORIES} "${_compiler_directory}")
    foreach(directory IN LISTS NTL_LIBRARY_DIRS)
        get_filename_component(prefix "${directory}" DIRECTORY)
        list(APPEND _runtime_directories "${directory}" "${prefix}/bin")
    endforeach()
    list(REMOVE_DUPLICATES _runtime_directories)

    if(MINGW)
        set(_runtime_tool objdump)
    else()
        set(_runtime_tool dumpbin)
    endif()
    find_program(NTL_RUNTIME_SCANNER NAMES ${_runtime_tool}
        HINTS "${_compiler_directory}" REQUIRED)
    configure_file(copy_runtime.cmake.in copy_runtime.cmake @ONLY)

    add_custom_target(ntl_runtime ALL
        COMMAND "${CMAKE_COMMAND}"
                "-DNTL_EXECUTABLE=$<TARGET_FILE:ntl_gf2x_gcd>"
                -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime.cmake"
        DEPENDS ntl_gf2x_gcd
        VERBATIM
    )
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME ntl_gcd_smoke
        COMMAND "${Python3_EXECUTABLE}" -m ntl.smoke_test
                --executable "$<TARGET_FILE:ntl_gf2x_gcd>"
    )
    set_tests_properties(ntl_gcd_smoke PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
        TIMEOUT 60
    )
    if(WIN32)
        set_tests_properties(ntl_gcd_smoke PROPERTIES
            ENVIRONMENT "PATH=$ENV{SystemRoot}\\System32"
        )
    endif()
endif()

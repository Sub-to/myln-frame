cmake_minimum_required(VERSION 3.14)
project(myln-frame VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MYLN_NATIVE         "Tune for this CPU (-march=native). Binaries won't run on older CPUs" OFF)
option(MYLN_BUILD_EXAMPLES "Build the example programs"  ON)
option(MYLN_BUILD_TESTS    "Build and register the tests (ctest)" ON)
option(MYLN_BUILD_BENCH    "Build the benchmark"          ON)
option(MYLN_SANITIZE       "Build with AddressSanitizer + UBSan (Linux/macOS)" OFF)

# ── ヘッダオンリーのコア（C++ から直接使う場合）─────────────────
# #include <myln/frame.h> / <myln/cascade.h> / <myln/heads/...> / <myln/tuner/...>
add_library(myln_headers INTERFACE)
add_library(myln::headers ALIAS myln_headers)
target_include_directories(myln_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(myln_headers INTERFACE cxx_std_17)
set_target_properties(myln_headers PROPERTIES EXPORT_NAME headers)

# ── 共通のコンパイルオプション ─────────────────────────────────
function(myln_configure target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra)
    if(MYLN_NATIVE)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
    if(MYLN_SANITIZE)
      target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endfunction()

# ── コアライブラリ（共有ライブラリ）──────────────────────────
# Python / Node.js / Ruby / Go / Rust など全言語から呼べる C API
add_library(myln SHARED bridge/myln_c_api.cpp)
add_library(myln::myln ALIAS myln)
target_link_libraries(myln PUBLIC myln_headers)
target_include_directories(myln PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bridge>)
target_compile_definitions(myln PRIVATE MYLN_VERSION_STRING="${PROJECT_VERSION}")
set_target_properties(myln PROPERTIES
    EXPORT_NAME myln
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    C_VISIBILITY_PRESET   hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
myln_configure(myln)

# ── デモ実行ファイル ──────────────────────────────────────
if(MYLN_BUILD_EXAMPLES)
  foreach(ex hello_myln chibitaru_demo)
    add_executable(${ex} examples/${ex}.cpp)
    target_link_libraries(${ex} PRIVATE myln_headers)
    myln_configure(${ex})
  endforeach()
endif()

# ── ベンチマーク ──────────────────────────────────────────
if(MYLN_BUILD_BENCH)
  add_executable(myln_bench bench/bench.cpp)
  target_link_libraries(myln_bench PRIVATE myln_headers)
  target_include_directories(myln_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  myln_configure(myln_bench)
endif()

# ── テスト ────────────────────────────────────────────────
if(MYLN_BUILD_TESTS)
  enable_testing()
  # C API もテストするので、共有ライブラリではなくソースを直接リンクする
  add_executable(myln_test_core tests/test_core.cpp bridge/myln_c_api.cpp)
  target_link_libraries(myln_test_core PRIVATE myln_headers)
  target_include_directories(myln_test_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} tests)
  target_compile_definitions(myln_test_core PRIVATE MYLN_VERSION_STRING="${PROJECT_VERSION}")
  myln_configure(myln_test_core)
  add_test(NAME core COMMAND myln_test_core)

  # Python ブリッジ（共有ライブラリ + ctypes）。python3 があれば登録する
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME build_myln_lib
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target myln --config $<CONFIG>)
    add_test(NAME python_bridge
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py)
    set_tests_properties(python_bridge PROPERTIES
        ENVIRONMENT "MYLN_LIB=$<TARGET_FILE:myln>")
    set_tests_properties(build_myln_lib PROPERTIES FIXTURES_SETUP myln_lib)
    set_tests_properties(python_bridge  PROPERTIES FIXTURES_REQUIRED myln_lib)
  endif()
endif()

# ── インストール（find_package(myln) 対応）─────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS myln myln_headers
        EXPORT mylnTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/myln DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY heads  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/myln FILES_MATCHING PATTERN "*.h")
install(DIRECTORY tuner  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/myln FILES_MATCHING PATTERN "*.h")
install(FILES bridge/myln_c_api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/myln)
install(EXPORT mylnTargets
        FILE mylnTargets.cmake
        NAMESPACE myln::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/myln)

configure_package_config_file(
    cmake/mylnConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/mylnConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/myln)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mylnConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mylnConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/mylnConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/myln)

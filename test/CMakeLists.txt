# --------------------------------------------------------------------------- #
#    CMake file for HQPBlock/test                                             #
#                                                                             #
#    In this file we define the unit tests for this library.                  #
#    Tests that should be supported by ctest executable must be added         #
#    with the add_test() command.                                             #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #

# Since we are using the block factory, objects from linked libraries
# may wrongly appear unused, and by default the linker does not include them,
# so we have to force the linking.
if (BUILD_SHARED_LIBS)
    if (UNIX AND (NOT APPLE))
        add_link_options("-Wl,--no-as-needed")
    endif ()
else ()
    if (MSVC)
        add_link_options("/WHOLEARCHIVE" "/FORCE:MULTIPLE")
    else () # Unix
        if (APPLE)
            add_link_options("-Wl,-all_load")
        else ()
            add_link_options("-Wl,--whole-archive,--allow-multiple-definition")
        endif ()
    endif ()
endif ()

# ----- HQPBlock_unit_test -------------------------------------------------- #

add_executable(${modName}_unit_test test.cpp)
target_link_libraries(${modName}_unit_test PRIVATE ${modNamespace}::${modName})

if (WIN32 AND BUILD_SHARED_LIBS AND NOT DEFINED ENV{CONDA_BUILD})
    # On Windows with shared libraries enabled copies all runtime DLL
    # dependencies next to the executable in the build directory
    # (POST_BUILD), using CMake's TARGET_RUNTIME_DLLS generator expression
    add_custom_command(TARGET ${modName}_unit_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${modName}_unit_test>
            $<TARGET_FILE_DIR:${modName}_unit_test>
            COMMAND_EXPAND_LISTS
            COMMENT "Copying runtime DLLs next to ${modName}_unit_test.exe")
    # one link at a time, since two copies of the same DLL into a
    # directory collide, and Ninja links in parallel
    get_property(_pools GLOBAL PROPERTY JOB_POOLS)
    if (NOT _pools MATCHES "smspp_runtime_dlls=")
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS smspp_runtime_dlls=1)
    endif ()
    set_property(TARGET ${modName}_unit_test
                 PROPERTY JOB_POOL_LINK smspp_runtime_dlls)
endif ()

add_test(NAME ${modName}_unit_test
         COMMAND ${modName}_unit_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# The label lets the module CI select exactly the tests exercising this
# module with `ctest -L <module>` (see tests/cmake/TestLabels.cmake in the
# umbrella for the system-tests counterpart).
set_tests_properties(${modName}_unit_test PROPERTIES LABELS "${modName}")

# HQPStageSolver on the instances of data/nc4 with the stage structure (two
# of them are in the system tests too), against the solution of each
# level that they carry, once the instances are downloaded and extracted
# [see the CMakeLists.txt of the module]; the lock is shared by the tests
# that build a target of the data of the module
add_test(NAME ${modName}_extract_nc4
         COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                 --target extract_${DATA_PREFIX}_nc4)
set_tests_properties(${modName}_extract_nc4 PROPERTIES
                     FIXTURES_SETUP ${modName}_nc4
                     RESOURCE_LOCK ${DATA_PREFIX}_data
                     LABELS "${modName}")

add_test(NAME ${modName}_unit_test_radial
         COMMAND ${modName}_unit_test
                 ../data/nc4/radial8_N010_s000.nc4
                 ../data/nc4/radial8_N010_s001.nc4
                 ../data/nc4/radial8_N010_s002.nc4
                 ../data/nc4/radial8_N010_s085.nc4
                 ../data/nc4/radial8_N010_s086.nc4
                 ../data/nc4/radial8_N020_s000.nc4
                 ../data/nc4/radial8_N020_s001.nc4
                 ../data/nc4/radial8_N020_s002.nc4
                 ../data/nc4/radial8_N020c_s000.nc4
                 ../data/nc4/radial8_N020c_s001.nc4
                 ../data/nc4/radial8_N020c_s002.nc4
                 ../data/nc4/radial8_N050_s000.nc4
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(${modName}_unit_test_radial PROPERTIES
                     FIXTURES_REQUIRED ${modName}_nc4
                     LABELS "${modName}")

# --------------------------------------------------------------------------- #

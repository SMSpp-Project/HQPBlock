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

# ----- Inner Solver -------------------------------------------------------- #
# The HQPSolver of the testers take their inner Solver from the factory, by
# the names in the configurations of this directory: MILPSolver provides
# them. In the umbrella it is a target (or will be, coming after this
# module), alone it is looked for among the installed packages; without it
# the testers check only what needs no inner Solver.
if (NOT (TARGET ${modNamespace}::MILPSolver OR BUILD_MILPSolver))
    find_package(MILPSolver QUIET)
endif ()

# ----- HQPBlock_unit_test -------------------------------------------------- #

add_executable(${modName}_unit_test test.cpp)
target_link_libraries(${modName}_unit_test PRIVATE ${modNamespace}::${modName})
if (TARGET ${modNamespace}::MILPSolver OR BUILD_MILPSolver)
    target_link_libraries(${modName}_unit_test PRIVATE
                          ${modNamespace}::MILPSolver)
endif ()

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
endif ()

add_test(NAME ${modName}_unit_test
         COMMAND ${modName}_unit_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# The label lets the module CI select exactly the tests exercising this
# module with `ctest -L <module>` (see tests/cmake/TestLabels.cmake in the
# umbrella for the system-tests counterpart).
set_tests_properties(${modName}_unit_test PROPERTIES LABELS "${modName}")

# ----- HQPBlock_radial_test ------------------------------------------------ #
# The hierarchical QPs of data/nc4, extracted by a test that sets up the
# fixture hqp_nc4_extracted [see the section Data of CMakeLists.txt].

add_executable(${modName}_radial_test test_radial.cpp)
target_link_libraries(${modName}_radial_test PRIVATE
                      ${modNamespace}::${modName})
if (TARGET ${modNamespace}::MILPSolver OR BUILD_MILPSolver)
    target_link_libraries(${modName}_radial_test PRIVATE
                          ${modNamespace}::MILPSolver)
endif ()

add_test(NAME ${modName}_radial_test
         COMMAND ${modName}_radial_test ${PROJECT_SOURCE_DIR}/data/nc4
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(${modName}_radial_test PROPERTIES
                     LABELS "${modName}" SKIP_RETURN_CODE 77
                     FIXTURES_REQUIRED hqp_nc4_extracted)

foreach (fmt ${DATA_FORMATS})
    add_test(NAME ${modName}_extract_hqp_${fmt}
             COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                     --target extract_hqp_${fmt})
    set_tests_properties(${modName}_extract_hqp_${fmt} PROPERTIES
                         FIXTURES_SETUP hqp_${fmt}_extracted
                         RESOURCE_LOCK hqp_data
                         LABELS "${modName}")
endforeach ()

# --------------------------------------------------------------------------- #

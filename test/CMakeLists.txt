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

# The classes of the modules register themselves in the factories from
# static objects, which nothing here names: a shared library is kept by
# what its module asks of the linker, while an archive has to be taken
# whole [see SMSppModule.cmake of SMS++].
smspp_keep_modules()

# ----- HQPBlock_unit_test -------------------------------------------------- #

add_executable(${modName}_unit_test test.cpp)
target_link_libraries(${modName}_unit_test PRIVATE ${modNamespace}::${modName})

smspp_copy_runtime(${modName}_unit_test)

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

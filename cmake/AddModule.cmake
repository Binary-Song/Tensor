#AddModule(
#   header_only/compiled     #[opt.]  whether the module is header-only or compiled. Must specify exactly one of them.
#   module_name     #[sing.] name of the module
#   dependencies    #[mult.] libraries the module depends on
#   src             #[mult.] source files, if not header-only
#   tests           #[mult.] test sources, each file built into one executable. Tests will depend on TEST_DEFAULT_DEPENDENCY if set. Ususally, you set TEST_DEFAULT_DEPENDENCY to Boost::unit_test_framework
#   test_dependencies #[mult.] By default, tests depend on: 1) the module being tested, 2) TEST_DEFAULT_DEPENDENCY. If these are not enough, you specify test_dependencies to add others.
#)
# 
function(AddModule)
    cmake_parse_arguments(
        "arg"  # prefix
        "header_only;compiled"  # optional args
        "module_name"   # one value args
        "dependencies;test_dependencies;src;tests"  # multi value args
        ${ARGN}
    )
    if(arg_header_only)
        message("[${arg_module_name}] header-only module")
    else()
        message("[${arg_module_name}] compiled module")
    endif()

    # Add main target
    if(arg_header_only)
        add_library(${arg_module_name} INTERFACE)
        target_include_directories(${arg_module_name} 
            INTERFACE 
                .
        )
        target_link_libraries(${arg_module_name} 
            INTERFACE 
                ${arg_dependencies}
        )
    elseif(arg_compiled)
        add_library(${arg_module_name}
            ${arg_src}
        )
        target_include_directories(${arg_module_name} 
            PUBLIC
                . 
        )
        target_link_libraries(${arg_module_name} 
            PUBLIC 
                ${arg_dependencies}
        )
    else()
        message(FATAL_ERROR "ERROR: MODULE [${arg_module_name}] NEEDS TO BE EITHER `header_only` OR `compiled`! SPECIFY ONE EXPLICITLY.")
    endif()
    message("- dependencies: ${arg_dependencies}")

    # Add test target
    if(BUILD_TESTING)
        foreach(test_file ${arg_tests})
            get_filename_component(test_file_no_ext  ${test_file}  NAME_WE)
            set(test_name "${arg_module_name}_Test_${test_file_no_ext}")
            add_executable("${test_name}" ${test_file})
            target_link_libraries("${test_name}"
                PRIVATE # Test public libs  
                    ${arg_module_name}
                    ${TEST_DEFAULT_DEPENDENCY}
                    ${arg_test_dependencies}
            )
            add_test(NAME "${test_name}" COMMAND "${test_name}")
        endforeach() 
        message("- test dependencies: ${arg_module_name} ${arg_test_dependencies} ${TEST_DEFAULT_DEPENDENCY}")

    endif() 
endfunction(AddModule)

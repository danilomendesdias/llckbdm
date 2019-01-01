add_library(_bindings MODULE _bindings.cpp)

target_link_libraries(_bindings PRIVATE pybind11::module)

set_target_properties(_bindings PROPERTIES
        PREFIX ""
        SUFFIX "${python_extension}")

set_target_properties(_bindings PROPERTIES OUTPUT_NAME _bindings)

if(SKBUILD)
    install(TARGETS _bindings LIBRARY DESTINATION llckbdm)
else()
    if(WIN32)
        set(python_extension ".pyd")
    elseif(APPLE)
        set(python_extension ".dynlib")
    else()
        set(python_extension ".so")
    endif()

    install(
        TARGETS _bindings
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/llckbdm
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/llckbdm
        ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/llckbdm
    )
endif()

add_library(_bindings MODULE _bindings.cpp)

if(DEFINED ENV{CONDA_PREFIX})
    target_link_libraries(
        _bindings PRIVATE
        pybind11::module
        Eigen3::Eigen
    )
else()
    target_link_libraries(
        _bindings PRIVATE
        pybind11::module
    )
endif()

set_target_properties(
    _bindings PROPERTIES
        PREFIX ""
)

if(SKBUILD)
    message("Scikit-build detected.")
    install(TARGETS _bindings LIBRARY DESTINATION llckbdm)
else()
    install(
        TARGETS _bindings
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/llckbdm
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/llckbdm
        ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/llckbdm
    )
endif()

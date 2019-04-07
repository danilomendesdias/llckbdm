# add_library(_kbdm SHARED kbdm.cpp)

if (TARGET Eigen3::Eigen)
    set(LINK_TARGETS Eigen3::Eigen)
endif()

if (MKL_FOUND)
    add_definitions(-DEIGEN_USE_LAPACKE)
    list(APPEND LINK_TARGETS ${MKL_LIBRARIES})
endif ()

# target_link_libraries(
#         _kbdm
#         ${LINK_TARGETS}
# )

pybind11_add_module(_bindings MODULE kbdm.cpp _bindings.cpp)

target_link_libraries(
        _bindings PRIVATE
        ${LINK_TARGETS}
)

if (WIN32)
    message("Windows detected. Forcing binding output to be .pyd.")
    set_target_properties(
            _bindings PROPERTIES
            PREFIX ""
            SUFFIX ".pyd"
    )
else()
    set_target_properties(
            _bindings PROPERTIES
            PREFIX ""
    )
endif()

message("Preparing targets to be inside: ${LLCKBDM_LIB_PATH}")

if(SKBUILD)
    message("Scikit-build detected.")
    install(TARGETS _bindings LIBRARY DESTINATION llckbdm)
#     install(TARGETS _kbdm LIBRARY DESTINATION ${LLCKBDM_LIB_PATH})
else()
    install(
            TARGETS _bindings
            RUNTIME DESTINATION ${LLCKBDM_LIB_PATH}
            LIBRARY DESTINATION ${LLCKBDM_LIB_PATH}
            ARCHIVE DESTINATION ${LLCKBDM_LIB_PATH}
    )

#     install(
#             TARGETS _kbdm
#             RUNTIME DESTINATION ${LLCKBDM_LIB_PATH}
#             LIBRARY DESTINATION ${LLCKBDM_LIB_PATH}
#             ARCHIVE DESTINATION ${LLCKBDM_LIB_PATH}    
#         )
endif()

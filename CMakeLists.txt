set(INC
  ..
)
set(INC_SYS
)

set(LIBRARIES
  cycles_device
  cycles_kernel
  cycles_render
  cycles_bvh
  cycles_subd
  cycles_graph
  cycles_util
)

if(WITH_CYCLES_OSL)
  list(APPEND LIBRARIES cycles_kernel_osl)
endif()

list(APPEND LIBRARIES ${CYCLES_GL_LIBRARIES} ${CYCLES_GLEW_LIBRARIES})

# Common configuration.

cycles_link_directories()

add_definitions(${GL_DEFINITIONS})

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

# Application build targets

if(WITH_CYCLES_MYAPP)
  set(SRC
    bake.cpp
  )
  add_executable(myapp ${SRC} ${INC} ${INC_SYS})
  unset(SRC)

  target_link_libraries(myapp ${LIBRARIES})
  cycles_target_link_libraries(myapp)

  if(UNIX AND NOT APPLE)
    set_target_properties(myapp PROPERTIES INSTALL_RPATH $ORIGIN/lib)
  endif()

  if(CYCLES_STANDALONE_REPOSITORY)
    cycles_install_libraries(myapp)
  endif()
endif()
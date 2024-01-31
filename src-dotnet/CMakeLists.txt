# Darknet object detection framework


# == 
# Darknet CLI is actually quite simple in terms of source files.
# ==
MESSAGE(STATUS "Setting up DARKNET DotNet")


IF (WIN32)
    INCLUDE_DIRECTORIES (src-dotnet)
    
    MESSAGE(STATUS "Setting up DarkentDotNet LIB")
    ADD_LIBRARY (darknetdotnet SHARED darknetdotnet.cpp)
    SET_TARGET_PROPERTIES (darknetdotnet PROPERTIES PUBLIC_HEADER "darknetdotnet.hpp")
    SET_TARGET_PROPERTIES (darknetdotnet PROPERTIES COMMON_LANGUAGE_RUNTIME "" COMMON_LANGUAGE_RUNTIME "netcore" DOTNET_TARGET_FRAMEWORK "net8.0")
    TARGET_LINK_LIBRARIES (darknetdotnet PRIVATE darknetobjlib)
    INSTALL (TARGETS darknetdotnet LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

ENDIF ()

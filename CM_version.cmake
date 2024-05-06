# Darknet object detection framework


# Create a version string from the git tag and commit hash (see src/darknet_version.h.in).
# Should look similar to this:
#
#		v1.99-63-gc5c3569
#
EXECUTE_PROCESS (COMMAND git describe --tags --dirty OUTPUT_VARIABLE DARKNET_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
MESSAGE (STATUS "Darknet ${DARKNET_VERSION_STRING}")

STRING (REGEX MATCH "v([0-9]+)\.([0-9]+)-([0-9]+)-g([0-9a-fA-F]+)" _ ${DARKNET_VERSION_STRING})
# note that MATCH_4 is not numeric

SET (DARKNET_VERSION_SHORT ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})

#message (STATUS "'${CMAKE_MATCH1}' '${CMAKE_MATCH2}' '${CMAKE_MATCH3}'")

#IF (DARKNET_VERSION_SHORT STREQUAL "..")
#    #SET (DARKNET_VERSION_SHORT "v2.0-143-g6fc77f49-dirty")
#    SET (DARKNET_VERSION_SHORT "v2.0-dirty")
#ENDIF()

##SET (DARKNET_VERSION_SHORT "v2.0-dirty")
#message (STATUS "Using version ${DARKNET_VERSION_SHORT}")

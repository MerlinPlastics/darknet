# Darknet object detection framework


FIND_PACKAGE (Threads REQUIRED)
MESSAGE (STATUS "Found Threads ${Threads_VERSION}")


FIND_PACKAGE (OpenCV CONFIG REQUIRED)
MESSAGE (STATUS "Found OpenCV ${OpenCV_VERSION}")
INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})
ADD_COMPILE_DEFINITIONS (DOPENCV) # TODO remove this


FIND_PACKAGE (OpenMP QUIET) # optional
IF (NOT OPENMP_FOUND)
	MESSAGE (WARNING "OpenMP not found. Building Darknet without support for OpenMP.")
ELSE ()
	MESSAGE (STATUS "Found OpenMP ${OpenMP_VERSION}")
	ADD_COMPILE_DEFINITIONS (DOPENMP)
	ADD_COMPILE_OPTIONS(-fopenmp)
#	TODO LDFLAGS+= -lgomp
ENDIF ()


set(SRCS kmeans.cpp distance.cpp cluster.cpp)
add_library(${PROJECT_NAME} SHARED ${SRCS})
add_library(${PROJECT_NAME}_static STATIC ${SRCS})

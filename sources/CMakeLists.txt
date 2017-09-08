
AUX_SOURCE_DIRECTORY(. SRC_LIST)

message(${SRC_LIST})

add_executable(ASJDetector ${SRC_LIST})
target_link_libraries(ASJDetector ${OpenCV_LIBS})
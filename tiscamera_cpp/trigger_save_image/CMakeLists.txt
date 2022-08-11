project(trigger_save_image)
 
cmake_minimum_required(VERSION 2.6)
set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
pkg_check_modules(TCAMLIB tcam)


include_directories( ${CMAKE_CURRENT_BINARY_DIR} ${GSTREAMER_INCLUDE_DIRS} ${TCAM_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_definitions(${GSTREAMER_CFLAGS_OTHER})  

add_executable(trigger_save_image main.cpp tcamcamera.cpp )
target_link_libraries(trigger_save_image ${TCAMLIB_LIBRARIES} ${GSTREAMER_LIBRARIES} ${OpenCV_LIBS})

add_executable(Bayer2RGB Bayer2RGB.cpp tcamcamera.cpp )
target_link_libraries(Bayer2RGB ${TCAMLIB_LIBRARIES} ${GSTREAMER_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS trigger_save_image RUNTIME DESTINATION bin)

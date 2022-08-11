project(calibration_save_image)
 
cmake_minimum_required(VERSION 2.6)
set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE Debug)


pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
pkg_check_modules(TCAMLIB tcam)


include_directories( ${CMAKE_CURRENT_BINARY_DIR} ${GSTREAMER_INCLUDE_DIRS} ${TCAM_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_definitions(${GSTREAMER_CFLAGS_OTHER})  

add_executable(calibration_save_image main.cpp tcamcamera.cpp )

target_link_libraries(calibration_save_image ${TCAMLIB_LIBRARIES} ${GSTREAMER_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS calibration_save_image RUNTIME DESTINATION bin)

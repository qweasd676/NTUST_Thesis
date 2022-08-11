////////////////////////////////////////////////////////////////////////
/* Using Property Example
This sample shows, how to set Gain and Expoures properties programmatically

It uses the the examples/cpp/common/tcamcamera.* files as wrapper around the
GStreamer code and property handling. Adapt the CMakeList.txt accordingly.

For some cameras, the automatic properties are available, when the camera's 
GStreamer pipeline is in READY state. Currently this is, after the live video 
has been started.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include <ctime> 
#include <time.h>
#include <chrono>
#include <thread>

#include "tcamcamera.h"
#include <unistd.h>
#include "opencv2/opencv.hpp"

#include <sys/types.h>
#include <sys/stat.h>

using namespace gsttcam;
using namespace std;


// Create a custom data structure to be passed to the callback function. 
typedef struct
{
    int ImageCounter;
    bool SaveNextImage;
    bool busy;
   	cv::Mat frame; 
} CUSTOMDATA;

#define B_OFFSET    0
#define G_OFFSET    1
#define R_OFFSET    2
#define ODD(i)      ((i)&1)
#define EVEN(i)     (!((i)&1))
#define IMAGE_HEIGHT 1080
#define IMAGE_WIDTH 1440

void BayerConversion(unsigned char *rawData, unsigned char *dst, int dst_w, int dst_h, int rawData_w)
{
    int x,y;
    int yWidth=0, yRawWidth=0, ym1RawWidth = 0, yp1RawWidth;
    int pix;
    int Red,Green,Blue;
    memset(dst,0,sizeof(unsigned char)*dst_w*dst_h*4);
    for (y = 1; y < dst_h - 1; y++)
    {
        yWidth += dst_w;
        yRawWidth += rawData_w;
        ym1RawWidth = yRawWidth - rawData_w;
        yp1RawWidth = yRawWidth + rawData_w;
        for (x = 1; x < dst_w - 1; x++)
        {
            pix = ((x+yWidth)<<2);
            if (ODD(y))
                if (EVEN(x))
                {   
                    Blue  = ((rawData[x-1+  yRawWidth] + 
                              rawData[x+1+  yRawWidth]) >> 1);
                    Green =   rawData[x  +  yRawWidth];
                    Red   = ((rawData[x  +ym1RawWidth] + 
                              rawData[x  +yp1RawWidth]) >> 1);
                }
                else
                {   // ODD(y) EVEN(x)
                    Blue  =   rawData[x  +  yRawWidth];
                    Green = ((rawData[x-1+  yRawWidth] + 
                              rawData[x+1+  yRawWidth] + 
                              rawData[x  +ym1RawWidth] + 
                              rawData[x  +yp1RawWidth]) >> 2);
                    Red   = ((rawData[x-1+ym1RawWidth] +
                              rawData[x+1+ym1RawWidth] +
                              rawData[x-1+yp1RawWidth] +
                              rawData[x+1+yp1RawWidth]) >> 2);
                }
            else
                if (EVEN(x))
                {   // EVEN(y) ODD(x)
                    Blue  = ((rawData[x-1+ym1RawWidth] +
                              rawData[x+1+ym1RawWidth] +
                              rawData[x-1+yp1RawWidth] +
                              rawData[x+1+yp1RawWidth]) >> 2);
                    Green = ((rawData[x-1+  yRawWidth] +
                              rawData[x+1+  yRawWidth] +
                              rawData[x  +ym1RawWidth] +
                              rawData[x  +yp1RawWidth]) >> 2);
                    Red   =   rawData[x  +  yRawWidth];
                }
                else
                {   //EVEN(y) EVEN(x)
                    Blue  = ((rawData[x  +ym1RawWidth] +
                              rawData[x  +yp1RawWidth]) >> 1);
                    Green =   rawData[x  +  yRawWidth];
                    Red   = ((rawData[x-1+  yRawWidth] +
                              rawData[x+1+  yRawWidth]) >> 1);
                }
            dst[pix+R_OFFSET] = Red;
            dst[pix+G_OFFSET] = Green;
            dst[pix+B_OFFSET] = Blue;
            // printf("%d %d %d\n", dst[pix+R_OFFSET], dst[pix+G_OFFSET], dst[pix+B_OFFSET]);
            // std::cout << dst[pix+R_OFFSET] << std::endl;
            // std::cout << dst[pix+G_OFFSET] << std::endl;
            // std::cout << dst[pix+B_OFFSET] << std::endl;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// List available properties helper function. Call this functions, if a
// list of available camera properties is needed.
void ListProperties(TcamCamera &cam)
{
    // Get a list of all supported properties and print it out
    auto properties = cam.get_camera_property_list();
    std::cout << "Properties:" << std::endl;
    for(auto &prop : properties)
    {
        std::cout << prop->to_string() << std::endl;
    }
}

chrono::steady_clock::time_point start;
char* cam_id;
bool begin_bool;

int writeFile(const char* _fileName, void* _buf, int _bufLen)
{
    FILE * fp = NULL;
    if( NULL == _buf || _bufLen <= 0 ) return (-1);

    fp = fopen(_fileName, "wb"); // 必須確保是以 二進位制寫入的形式開啟

    if( NULL == fp )
    {
        return (-1);
    }

    fwrite(_buf, _bufLen, 1, fp); //二進位制寫

    fclose(fp);
    fp = NULL;

    return 0;    
}

int count_image = 0;
void save(unsigned char* buf, int len, int imageCounter){
    char ImageFileName[256];
    sprintf(ImageFileName, "%s/image%05d.jpg", cam_id, imageCounter);
    cv::Mat image_bgr;
    // image_bgr.create(1080,1440,CV_8UC(1));
    // memcpy(image_bgr.data, buf, 1440*1080*1);
    
    image_bgr.create(IMAGE_HEIGHT,IMAGE_WIDTH,CV_8UC(1));
    memcpy( image_bgr.data, buf, IMAGE_WIDTH*IMAGE_HEIGHT*1);
    // BayerConversion(buf, image_bgr.data, 1440, 1080, 1440);
    cv::imwrite(ImageFileName,image_bgr);
    free(buf);
    image_bgr.release();
    count_image--;
}

GstFlowReturn new_frame_cb(GstAppSink *appsink, gpointer data)
{
    int width, height ;
    const GstStructure *str;

    if(begin_bool){
        begin_bool = false;
        start = chrono::steady_clock::now();
    }
    // Cast gpointer to CUSTOMDATA*
    CUSTOMDATA *pCustomData = (CUSTOMDATA*)data;

    if( !pCustomData->SaveNextImage){
        std::cout << "No!!" << std::endl;
        return GST_FLOW_OK;
    }
    pCustomData->SaveNextImage = false;

    if (pCustomData->ImageCounter != 0 && pCustomData->ImageCounter % 100 == 0) // 取固定幀數為100幀
    {
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(end-start);

        float cost = time_used.count();
        float fps = (float)(pCustomData->ImageCounter) / cost;
        std::cout << "cost: " << cost << std::endl;
        std::cout << "fps: " << fps << std::endl;
        std::cout << "count: " <<count_image <<std::endl;
    }
    // The following lines demonstrate, how to acces the image
    // data in the GstSample.
    GstSample *sample = gst_app_sink_pull_sample(appsink);

    GstBuffer *buffer = gst_sample_get_buffer(sample);

    GstMapInfo info;

    gst_buffer_map(buffer, &info, GST_MAP_READ);
    
    if (info.data != NULL) 
    {
        // info.data contains the image data as blob of unsigned char 

        GstCaps *caps = gst_sample_get_caps(sample);
        // Get a string containg the pixel format, width and height of the image        
        str = gst_caps_get_structure (caps, 0);    

        if( strcmp( gst_structure_get_string (str, "format"),"rggb") == 0)  
        {
            // Now query the width and height of the image
            gst_structure_get_int (str, "width", &width);
            gst_structure_get_int (str, "height", &height);

            // Create a cv::Mat, copy image data into that and save the image.
            unsigned char *buffer = (unsigned char *)malloc(height*width*1*sizeof(unsigned char));
            memcpy( buffer, info.data, width*height*1);
            std::thread t1(save, buffer, width*height*1, pCustomData->ImageCounter);
            t1.detach();
            count_image++;
        }

    }
    pCustomData->ImageCounter++;
    // Calling Unref is important!
    gst_buffer_unmap (buffer, &info);
    gst_sample_unref(sample);
    pCustomData->SaveNextImage = true;

    // Set our flag of new image to true, so our main thread knows about a new image.
    return GST_FLOW_OK;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    gst_init(&argc, &argv);

    // Declaration of the Pointers to the properties.
    std::shared_ptr<Property> ExposureAuto = NULL;
    std::shared_ptr<Property> ExposureValue = NULL;
    std::shared_ptr<Property> GainAuto = NULL;
    std::shared_ptr<Property> GainValue = NULL;
    std::shared_ptr<Property> WhitebalanceAuto = NULL;
    std::shared_ptr<Property> TriggerMode = NULL;

    CUSTOMDATA CustomData;
    CustomData.ImageCounter = 0;
    CustomData.SaveNextImage = true;
    begin_bool = true;
    cam_id = argv[1];

    int check = mkdir(cam_id,0777);
    // check if directory is created or not
    if (!check)
        printf("Directory created\n");
    else {
        printf("Unable to create directory\n");
    }

    // Initialize our TcamCamera object "cam" with the serial number
    // of the camera, which is to be used in this program.
    TcamCamera cam(cam_id);
    //TcamCamera cam("00001234");
    
    // Set a color video format, resolution and frame rate
    cam.set_capture_format("rggb", FrameSize{IMAGE_WIDTH,IMAGE_HEIGHT}, FrameRate{2500000,10593});

    // Comment following line, if no live video display is wanted.
    //cam.enable_video_display(gst_element_factory_make("ximagesink", NULL));
    
    // Register a callback to be called for each new frame
    cam.set_new_frame_callback(new_frame_cb, &CustomData);

    // Uncomment following line, if properties shall be listed. Sometimes
    // the property names of USB and GigE cameras differ.
    // ListProperties(cam);

    // Query the pointers to the white balance properties. If a properties
    // does not exist an exception is thrown.
    try
    {
        ExposureAuto = cam.get_property("Exposure Auto");
    }
    catch(std::exception &ex)    
    {
        printf("Error %s : %s\n",ex.what(), "Exposure Automatic");
    }

    try
    {
        ExposureValue = cam.get_property("Exposure Time (us)");
    }
    catch(std::exception &ex)    
    {
        printf("Error %s : %s\n",ex.what(), "Exposure Value");
    }

    try
    {
        GainAuto = cam.get_property("Gain Auto");
    }
    catch(std::exception &ex)    
    {
        printf("Error %s : %s\n",ex.what(), "Gain Automatic");
    }

    try
    {
        GainValue = cam.get_property("Gain");
    }
    catch(std::exception &ex)    
    {
        printf("Error %s : %s\n",ex.what(), "Gain Value");
    }

    try
    {
        WhitebalanceAuto = cam.get_property("Whitebalance Auto");
    }
    catch(std::exception &ex)    
    {
        printf("Error %s : %s\n",ex.what(), "Whitebalance Automatic");
    }

    try
    {
        TriggerMode = cam.get_property("Trigger Mode");
    }
    catch(std::exception &ex)    
    {
        printf("Error %s : %s\n",ex.what(), "Trigger Mode");
    }

    // Now get the current property values:
    if( ExposureAuto != NULL)
    {
        int Auto;
        ExposureAuto->get(cam,Auto);
        if( Auto == 1)
            printf("Current exposure automatic is On.\n");
        else
            printf("Current exposure automatic is Off.\n");
    }

    if( ExposureValue != NULL)
    {
        int ExposureTime;
        ExposureValue->get(cam,ExposureTime);
        printf("Current exposure time is %d.\n",ExposureTime);
    }

    if( GainAuto != NULL)
    {
        int Auto;
        GainAuto->get(cam,Auto);
        if( Auto == 1)
            printf("Current gain automatic is On.\n");
        else
            printf("Current gain  automatic is Off.\n");
    }

    if( GainValue != NULL)
    {
        int gain;
        GainValue->get(cam,gain);
        printf("Current gain value is %d.\n",gain);
    }

    if( WhitebalanceAuto != NULL)
    {
        int Auto;
        WhitebalanceAuto->get(cam,Auto);
        if( Auto == 1)
            printf("Current Whitebalance automatic is On.\n");
        else
            printf("Current Whitebalance automatic is Off.\n");
    }

    if( TriggerMode != NULL)
    {
        int Auto;
        TriggerMode->get(cam,Auto);
        if( Auto == 1)
            printf("Current Trigger mode is On.\n");
        else
            printf("Current Trigger mode is Off.\n");
    }

    // Disable automatics, so the property values can be set 
    // if( ExposureValue != NULL){
    //     ExposureValue->set(cam,500);
    // }
    // if( GainValue != NULL){3
    //     GainValue->set(cam,0);
    // }

    if( ExposureValue != NULL)
    {
        int ExposureTime;
        ExposureValue->get(cam,ExposureTime);
        printf("Current exposure time is %d.\n",ExposureTime);
    }

    if( ExposureAuto != NULL){
        ExposureAuto->set(cam,0);
    }

    if( GainAuto != NULL){
        GainAuto->set(cam,0);
    }

    if( WhitebalanceAuto != NULL){
        WhitebalanceAuto->set(cam,0);
    }

    if( TriggerMode != NULL){
        TriggerMode->set(cam,1);
    }
    if( TriggerMode != NULL)
    {
        int Auto;
        TriggerMode->get(cam,Auto);
        if( Auto == 1)
            printf("Current Trigger mode is On.\n");
        else
            printf("Current Trigger mode is Off.\n");
    }
    // set a value
    /*if( ExposureValue != NULL){
        ExposureValue->set(cam,333);
    }

    if( GainValue != NULL){
        GainValue->set(cam,400);
    }*/

    // Start the camera
    cam.start();
    std::cout << "No!!" << std::endl;
    while(1){
        sleep(1);
    }
    std::cout << "No!!" << std::endl;
    printf("Press enter key to end program.");

    // Simple implementation of "getch()", wait for enter key.
    char dummyvalue[10];
    scanf("%c",dummyvalue);

    cam.stop();

    return 0;
}

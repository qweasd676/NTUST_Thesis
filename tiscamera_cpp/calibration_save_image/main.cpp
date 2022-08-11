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
    char* local;
   	cv::Mat frame; 
} CUSTOMDATA;

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
char *cam1_id, *cam2_id;
bool begin_bool;
bool now_run;
GstFlowReturn new_frame_cb(GstAppSink *appsink, gpointer data)
{
    if(now_run) return GST_FLOW_OK;
    now_run = true;
    int width, height ;
    const GstStructure *str;

    if(begin_bool){
        begin_bool = false;
        start = chrono::steady_clock::now();
    }
    // Cast gpointer to CUSTOMDATA*
    CUSTOMDATA *pCustomData = (CUSTOMDATA*)data;
    /*if( !pCustomData->SaveNextImage){
        return GST_FLOW_OK;
    }*/
    
    //pCustomData->ImageCounter++;
    /*if (pCustomData->ImageCounter % 100 == 0) // 取固定幀數為100幀
    {
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(end-start);

        float cost = time_used.count();
        float fps = (float)(pCustomData->ImageCounter) / cost;
        std::cout << "cost: " << cost << std::endl;
        std::cout << "fps: " << fps << std::endl;
    }*/
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

        if( strcmp( gst_structure_get_string (str, "format"),"BGRx") == 0)  
        {
            // Now query the width and height of the image
            gst_structure_get_int (str, "width", &width);
            gst_structure_get_int (str, "height", &height);

            // Create a cv::Mat, copy image data into that and save the image.
            pCustomData->frame.create(height,width,CV_8UC(4));
            memcpy( pCustomData->frame.data, info.data, width*height*4);
            char ImageFileName[256];
            sprintf(ImageFileName, "%s/image%05d.jpg", pCustomData->local, pCustomData->ImageCounter);
            //cv::Mat gray;
            //cv::cvtColor(pCustomData->frame, gray, cv::COLOR_RGBA2GRAY);
            cv::Mat small;
            cv::resize(pCustomData->frame, small, cv::Size(600, 400), 0, 0, cv::INTER_LINEAR);
            cv::imshow(pCustomData->local, small);
            cv::waitKey(1);

            if( pCustomData->SaveNextImage)
            {
                pCustomData->ImageCounter++;
                cv::imwrite(ImageFileName,pCustomData->frame);
                pCustomData->SaveNextImage = false;
            }
        }

    }
    
    // Calling Unref is important!
    gst_buffer_unmap (buffer, &info);
    gst_sample_unref(sample);
    now_run = false;
    //pCustomData->SaveNextImage = true;

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

    now_run = false;
    int camera_total = argc-1;
    char** cam_id = argv + 1;
    CUSTOMDATA CustomData[camera_total];
    TcamCamera* cam[camera_total];
    for(int i = 0; i < camera_total; ++i){
        CustomData[i].ImageCounter = 0;
        CustomData[i].SaveNextImage = false;
        CustomData[i].local = cam_id[i];
        
        int check = mkdir(cam_id[i],0777);
        // check if directory is created or not
        if (!check)
            printf("%s Directory created\n", cam_id[i]);
        else {
            printf("%s Unable to create directory\n", cam_id[i]);
            //exit(1);
        }

        cam[i] = new TcamCamera(cam_id[i]);

        // Set a color video format, resolution and frame rate
        cam[i]->set_capture_format("BGRx", FrameSize{1440,1080}, FrameRate{2500000,10593});

        // Comment following line, if no live video display is wanted.
        //cam1.enable_video_display(gst_element_factory_make("ximagesink", NULL));
        
        // Register a callback to be called for each new frame
        cam[i]->set_new_frame_callback(new_frame_cb, &(CustomData[i]));

        // Uncomment following line, if properties shall be listed. Sometimes
        // the property names of USB and GigE cam1eras differ.
        // ListProperties(cam1);

        // Query the pointers to the white balance properties. If a properties
        // does not exist an exception is thrown.
        try
        {
            ExposureAuto = cam[i]->get_property("Exposure Auto");
        }
        catch(std::exception &ex)    
        {
            printf("Error %s : %s\n",ex.what(), "Exposure Automatic");
        }

        try
        {
            ExposureValue = cam[i]->get_property("Exposure Time (us)");
        }
        catch(std::exception &ex)    
        {
            printf("Error %s : %s\n",ex.what(), "Exposure Value");
        }

        try
        {
            GainAuto = cam[i]->get_property("Gain Auto");
        }
        catch(std::exception &ex)    
        {
            printf("Error %s : %s\n",ex.what(), "Gain Automatic");
        }

        try
        {
            GainValue = cam[i]->get_property("Gain");
        }
        catch(std::exception &ex)    
        {
            printf("Error %s : %s\n",ex.what(), "Gain Value");
        }

        try
        {
            WhitebalanceAuto = cam[i]->get_property("Whitebalance Auto");
        }
        catch(std::exception &ex)    
        {
            printf("Error %s : %s\n",ex.what(), "Whitebalance Automatic");
        }

        try
        {
            TriggerMode = cam[i]->get_property("Trigger Mode");
        }
        catch(std::exception &ex)    
        {
            printf("Error %s : %s\n",ex.what(), "Trigger Mode");
        }

        // Now get the current property values:
        if( ExposureAuto != NULL)
        {
            int Auto;
            ExposureAuto->get((*cam[i]),Auto);
            if( Auto == 1)
                printf("Current exposure automatic is On.\n");
            else
                printf("Current exposure automatic is Off.\n");
        }

        if( ExposureValue != NULL)
        {
            int ExposureTime;
            ExposureValue->get((*cam[i]),ExposureTime);
            printf("Current exposure time is %d.\n",ExposureTime);
        }

        if( GainAuto != NULL)
        {
            int Auto;
            GainAuto->get((*cam[i]),Auto);
            if( Auto == 1)
                printf("Current gain automatic is On.\n");
            else
                printf("Current gain  automatic is Off.\n");
        }

        if( GainValue != NULL)
        {
            int gain;
            GainValue->get((*cam[i]),gain);
            printf("Current gain value is %d.\n",gain);
        }

        if( WhitebalanceAuto != NULL)
        {
            int Auto;
            WhitebalanceAuto->get((*cam[i]),Auto);
            if( Auto == 1)
                printf("Current Whitebalance automatic is On.\n");
            else
                printf("Current Whitebalance automatic is Off.\n");
        }

        if( TriggerMode != NULL)
        {
            int Auto;
            TriggerMode->get((*cam[i]),Auto);
            if( Auto == 1)
                printf("Current Trigger mode is On.\n");
            else
                printf("Current Trigger mode is Off.\n");
        }

        // Disable automatics, so the property values can be set 
        if( ExposureAuto != NULL){
            ExposureAuto->set((*cam[i]),1);
        }

        if( GainAuto != NULL){
            GainAuto->set((*cam[i]),1);
        }

        if( WhitebalanceAuto != NULL){
            WhitebalanceAuto->set((*cam[i]),1);
        }

        if( TriggerMode != NULL){
            TriggerMode->set((*cam[i]),0);
        }


        cam[i]->start();
    }
    
    begin_bool = true;

    printf("Press any key to save image.");

    int index = 0;
    while(1){
        char dummyvalue[10];
        scanf("%c",dummyvalue);
        for(int i = 0; i < camera_total; ++i){
            CustomData[i].SaveNextImage = true;
        }
        std::cout << "save image " << index << std::endl;
        index++;
    }


    return 0;
}
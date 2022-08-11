#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include <ctime> 
#include <time.h>
#include <chrono>
#include <thread>
#include <dirent.h>

#include "tcamcamera.h"
#include <unistd.h>
#include "opencv2/opencv.hpp"

#include <sys/types.h>
#include <sys/stat.h>

using namespace gsttcam;
using namespace std;

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

int main(int argc, char **argv)
{

    char ImageFileName[256];
    char* cam_id = argv[1];

    int file_count = 0;
    DIR * dirp;
    struct dirent * entry;
    
    dirp = opendir(cam_id);

    while((entry = readdir(dirp)) != NULL){

        if(entry -> d_type == DT_REG){
            ++file_count;
        }
    }
    // closedir(cam_id);

    for(int i = 0; i < file_count; ++i){
        sprintf(ImageFileName, "%s/image%05d.jpg", cam_id, i);
        std::cout << ImageFileName << std::endl;
        cv::Mat image_rggb = cv::imread(ImageFileName, cv::IMREAD_UNCHANGED);
        // cv::imshow("1", image_rggb);
        // cv::waitKey(0);
        cv::Mat image_bgr;
        image_bgr.create(IMAGE_HEIGHT,IMAGE_WIDTH,CV_8UC(4));
        printf("%d %d\n", image_rggb.rows, image_rggb.cols);
        BayerConversion(image_rggb.data, image_bgr.data, 1440, 1080, 1440);
        cv::imwrite(ImageFileName,image_bgr);
    }
    
}
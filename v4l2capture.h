#ifndef _V4L2CAPTURE_H_
#define _V4L2CAPTURE_H_
#include <iostream>
#include <string>
#include <linux/videodev2.h>
#include "sysio.h"

using namespace std;


class V4L2Capture{
public:
    V4L2Capture();
    bool initialize();
private:
    string devname;
    int fd;
    unsigned int pixfmt;
    unsigned int width;
    unsigned int height;
    // this deinterlace fd used for possible deinterlacing
    // if video is progressive this contain last fd of g_buff 
    int deinterlace_buf_fd;
};

#endif
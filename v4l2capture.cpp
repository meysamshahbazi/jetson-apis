#include "v4l2capture.h"

V4L2Capture::V4L2Capture() 
{
    devname = "/dev/video0";
    width = 1920;
    height = 1080;
    pixfmt = V4L2_PIX_FMT_YUYV;
    fd = -1;
}

bool V4L2Capture::initialize()
{
    struct v4l2_format fmt;
    fd = open( devname.c_str() , O_RDWR | O_NONBLOCK );
    if (fd == -1){
        cout<<"Failed to open camera device: "<<strerror(errno)<< ", "<< errno<<endl;
        return false;
    }
    /* Set camera output format */
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    while (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        if(EBUSY == errno) { // TODO: handle this better!
            if (fd > 0)
                close(fd);

            fd = open(devname.c_str(), O_RDWR  | O_NONBLOCK );
            if (fd == -1){
                cout<<"Failed to open camera device: "<<strerror(errno)<< ", "<< errno<<endl;
                return false;
            }
        }
    }

    /* Get the real format in case the desired is not supported */
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
        cout<<"Failed to get camera output format: "<<strerror(errno)<< ", "<< errno<<endl;
        return false;
    }

    if (fmt.fmt.pix.width != width ||
        fmt.fmt.pix.height != height ||
        fmt.fmt.pix.pixelformat != pixfmt)
    {
        cout<<"The desired format is not supported for node:"<<devname<<endl;
        width = fmt.fmt.pix.width;
        height = fmt.fmt.pix.height;
        pixfmt =fmt.fmt.pix.pixelformat;
    }

    cout<<"Camera ouput format: "<<fmt.fmt.pix.width <<"x"<<fmt.fmt.pix.height
        <<" stride: "<<fmt.fmt.pix.bytesperline
        <<" imagesize: "<<fmt.fmt.pix.sizeimage<<endl;
             
    return true;
}

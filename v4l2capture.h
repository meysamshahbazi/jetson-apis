#ifndef _V4L2CAPTURE_H_
#define _V4L2CAPTURE_H_

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>



class V4L2Capture{
public:
    V4L2Capture();
private:
    char* devname;
    int fd;
    unsigned int pixfmt;
    unsigned int width;
    unsigned int height;
};

#endif
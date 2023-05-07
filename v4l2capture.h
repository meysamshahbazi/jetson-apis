#ifndef _V4L2CAPTURE_H_
#define _V4L2CAPTURE_H_
#include <iostream>
#include <string>
#include <linux/videodev2.h>
#include <pthread.h>
#include <map>

#include "sysio.h"
#include "nvbuf_utils.h"

// #include "tegra_drm.h"
#ifndef DOWNSTREAM_TEGRA_DRM
#include "tegra_drm_nvdc.h"
#endif
#include "NvApplicationProfiler.h"

using namespace std;

const static map<unsigned int, NvBufferColorFormat> nv_color_fmt {
        /* TODO: add more pixel format mapping */
        {V4L2_PIX_FMT_UYVY, NvBufferColorFormat_UYVY},
        {V4L2_PIX_FMT_VYUY, NvBufferColorFormat_VYUY},
        {V4L2_PIX_FMT_YUYV, NvBufferColorFormat_YUYV},
        {V4L2_PIX_FMT_YVYU, NvBufferColorFormat_YVYU},
        {V4L2_PIX_FMT_GREY, NvBufferColorFormat_GRAY8},
        {V4L2_PIX_FMT_YUV420M, NvBufferColorFormat_YUV420},
    };


class V4L2Capture{
    
public:
    V4L2Capture();
    ~V4L2Capture();
    bool initialize();
    bool prepare_buffers();
    bool request_camera_buff();
    bool isInterleave();
    bool start_stream();
    bool start_capture();
    bool grab_frame();
    bool TestCapture();
    static void* func_grab_thread(void* arg);
    static void* func_drm_render(void* arg);
private:
    const static int V4L2_BUFFERS_NUM {4};
    string devname;
    int cam_fd;
    unsigned int pixfmt;
    unsigned int width;
    unsigned int height;
    // this deinterlace fd used for possible deinterlacing
    // if video is progressive this contain last fd of g_buff 
    int deinterlace_buf_fd;

    /* User accessible pointer */
    unsigned char * mm_start[V4L2_BUFFERS_NUM];
    /* Buffer length */
    unsigned int buff_size[V4L2_BUFFERS_NUM];
    /* File descriptor of NvBuffer */
    int dmabuff_fd[V4L2_BUFFERS_NUM];

    bool quit{false};
    struct v4l2_buffer v4l2_buf;

    pthread_t ptid_drm;
    pthread_t ptid_grab;
    ///------------------------------------------------------------------------
    // for debug only !
};

#endif
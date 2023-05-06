#include "v4l2capture.h"
#include <chrono>
#include "NvDrmRenderer.h"

/**
 * @brief Construct a new V4L2Capture::V4L2Capture object
 * 
 */
V4L2Capture::V4L2Capture() 
{
    devname = "/dev/video0";
    width = 1920;
    height = 1080;
    pixfmt = V4L2_PIX_FMT_UYVY;
    cam_fd = -1;
}

/**
 * @brief Destroy the V4L2Capture::V4L2Capture object
 * 
 */
V4L2Capture::~V4L2Capture() {
    if (cam_fd > 0)
        close(cam_fd);
    for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++) {
        if (dmabuff_fd[i])
            NvBufferDestroy(dmabuff_fd[i]);
    }
}

/**
 * @brief 
 * 
 * @return true 
 * @return false 
 */
bool V4L2Capture::initialize()
{
    // open the camera device 
    cam_fd = open( devname.c_str() , O_RDWR | O_NONBLOCK );
    if (cam_fd == -1){
        cout<<"Failed to open camera device: "<<strerror(errno)<< ", "<< errno<<endl;
        return false;
    }
    // Set camera output format 
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if ( xioctl(cam_fd, VIDIOC_S_FMT, &fmt) < 0 ){
        cout<<"Failed to set camera format: "<<strerror(errno)<< ", "<< errno<<endl;
        return false;    
    }
    // Get the real format in case the desired is not supported 
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cam_fd, VIDIOC_G_FMT, &fmt) < 0) {
        cout<<"Failed to get camera output format: "<<strerror(errno)<< ", "<< errno<<endl;
        return false;
    }
    // check consistency between setted paramtaer and real paramters
    if (fmt.fmt.pix.width != width ||
        fmt.fmt.pix.height != height ||
        fmt.fmt.pix.pixelformat != pixfmt ) {
        cout<<"The desired format is not supported for node:"<<devname<<endl;
        width = fmt.fmt.pix.width;
        height = fmt.fmt.pix.height;
        pixfmt = fmt.fmt.pix.pixelformat;
    }
    // log camera info
    cout<< "Camera ouput format: " << fmt.fmt.pix.width << "x" <<fmt.fmt.pix.height
        << " stride: " << fmt.fmt.pix.bytesperline
        << " imagesize: " << fmt.fmt.pix.sizeimage << endl;
             
    return true;
}

bool V4L2Capture::prepare_buffers()
{
    NvBufferCreateParams input_params = {0};
    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = width;
    input_params.height = height;
    input_params.layout = NvBufferLayout_Pitch;
    input_params.colorFormat = nv_color_fmt.at(pixfmt);
    input_params.nvbuf_tag = NvBufferTag_CAMERA;
    /* Create buffer and provide it with camera */
    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++) {
        int fd;
        NvBufferParams params = {0};
        if (-1 == NvBufferCreateEx(&fd, &input_params)) {
            cout<<"Failed to create NvBuffer"<<endl;
            return false;
        }

        dmabuff_fd[index] = fd;

        if (-1 == NvBufferGetParams(fd, &params)){
            cout<<"Failed to get NvBuffer parameters"<<endl;
            return false;
        }

        if (-1 == NvBufferMemMap( dmabuff_fd[index], 0, NvBufferMem_Read_Write, // TODO: check for usefulness of this
                    (void**)&mm_start[index])) {
            cout<<"Failed to map buffer"<<endl;
            return false;
        }
    }

    if ( !request_camera_buff() ) {
        cout<<"Failed to set up camera buff"<<endl;
        return false;
    }

    if( isInterleave() ) {
        input_params.payloadType = NvBufferPayload_SurfArray;
        input_params.width = 720; // 736
        input_params.height = 576;
        input_params.layout = NvBufferLayout_Pitch;
        input_params.colorFormat = nv_color_fmt.at(pixfmt);
        input_params.nvbuf_tag = NvBufferTag_NONE;
        if (-1 == NvBufferCreateEx(&deinterlace_buf_fd, &input_params))
            cout<<"Failed to create NvBuffer deinterlace_buf_fd"<<endl;
            return false;
    }
    cout<<"Succeed in preparing stream buffers for camera "<<devname<<endl;
    return true;
}

bool V4L2Capture::start_stream()
{
    enum v4l2_buf_type type;
    /* Start v4l2 streaming */
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cam_fd, VIDIOC_STREAMON, &type) < 0) {
        cout<<"Failed to start streaming: "<<strerror(errno)<< ", "<< errno<<endl;
        return false;
    }
    usleep(200);
    cout<<"Camera video streaming on ..."<<endl;
    return true;   
}

bool V4L2Capture::isInterleave() {
    return height == 288;
}

bool V4L2Capture::request_camera_buff()
{
    /* Request camera v4l2 buffer */
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;

    if (xioctl(cam_fd, VIDIOC_REQBUFS, &rb) < 0) { // TODO : handle  errno == EBUSY
        cout<<"Failed to request v4l2 buffers:"<<strerror(errno)<< ", "<< errno<<endl;
        return false;
    }

    if (rb.count != V4L2_BUFFERS_NUM){
        cout<<"V4l2 buffer number is not as desired"<<endl;
        return false;
    }

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++) {
        struct v4l2_buffer buf;
        /* Query camera v4l2 buf length */
        memset(&buf, 0, sizeof(buf) );
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;
        if (xioctl(cam_fd, VIDIOC_QUERYBUF, &buf) < 0){
            cout<<"Failed to query buff: "<<strerror(errno)<< ", "<< errno<<endl;
            return false;
        }
        /* TODO: add support for multi-planer
           Enqueue empty v4l2 buff into camera capture plane */
        buf.m.fd = (unsigned long) dmabuff_fd[index];
        
        if( buf.length != buff_size[index]) {
            buff_size[index] = buf.length;
        }

        if (xioctl(cam_fd, VIDIOC_QBUF, &buf) < 0){
            cout<<"Failed to enqueue buffers: "<<strerror(errno)<< ", "<< errno<<endl;
            return false;
        }
    }

    return true;
}


bool V4L2Capture::grab_frame()
{
    /* Dequeue a camera buff */
    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4l2_buf.memory = V4L2_MEMORY_DMABUF;

    if (ioctl(cam_fd, VIDIOC_DQBUF, v4l2_buf) < 0) {
        cout<<"Failed to dequeue camera buff: "<<strerror(errno)<< ", "<< errno<<endl;  
        return false;
    }

    /* Cache sync for VIC operation since the data is from CPU */
    NvBufferMemSyncForDevice(dmabuff_fd[v4l2_buf.index], 0,
            (void**) &mm_start[v4l2_buf.index] );
    
    return true;
}


bool V4L2Capture::start_capture()
{
    NvBufferTransformParams transParams;
    /* Init the NvBufferTransformParams */
    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Bilinear;
    
    struct pollfd fds[1];   
    fds[0].events = POLLIN;

    while (!quit) {
        fds[0].fd = cam_fd;
        poll(fds, 1, 5000);
        // printf("im in theread grab\n");
        if ( (fds[0].revents & POLLIN)  /* && ctx->buf_fpga[getFpgaBufIndex(index)] != VID_DISCONNECTED   && ctx->changing_state[index] == 0 */) {
            auto begin = std::chrono::steady_clock::now();
            if(!grab_frame()) continue;
            auto end = std::chrono::steady_clock::now();
            // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
            std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

            deinterlace_buf_fd = dmabuff_fd[v4l2_buf.index];
            
            // if (-1 == NvBufferTransform(deinterlace_buf_fd, ctx->full_hd_fd[index], &transParams)) 
                    // printf("Failed to convert the buffer to full_hd_fd[i]");
            
            /* Enqueue camera buffer back to driver */    
            xioctl(cam_fd, VIDIOC_QBUF, &v4l2_buf);
        }
    }
}

bool V4L2Capture::TestCapture()
{
    struct drm_tegra_hdr_metadata_smpte_2086 metadata;
    NvDrmRenderer *drm_renderer;
    drm_renderer = NvDrmRenderer::createDrmRenderer("renderer0",
            1920, 1080, 0, 0,
            0, 0, metadata, 0);


    ctx->drm_renderer->setFPS(30);
}


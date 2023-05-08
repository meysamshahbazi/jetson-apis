#include "v4l2capture.h"
#include <chrono>
#include "NvDrmRenderer.h"
#include "cudaDeinterlace.h"

// for test



/**
 * @brief Construct a new V4L2Capture::V4L2Capture object
 * 
 */
V4L2Capture::V4L2Capture() 
{
    devname = "/dev/video0";
    width = 736;
    height = 288;
    // the format V4L2_PIX_FMT_UYVY dosnt suppurted with cuGraphicsEGLRegisterImage yet!
    // https://forums.developer.nvidia.com/t/uyvy-for-cugraphicseglregisterimage-in-32-2-sdk/78634/9
    pixfmt = V4L2_PIX_FMT_YUYV; 
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
    cam_fd = open(devname.c_str(), O_RDWR | O_NONBLOCK);
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
    memset(&v4l2_buf, 0, sizeof v4l2_buf );
    v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    v4l2_buf.memory = V4L2_MEMORY_DMABUF;

    if (ioctl(cam_fd, VIDIOC_DQBUF, &v4l2_buf) < 0) {
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
    typedef void * (*THREADFUNCPTR)(void *);
    
    pthread_create(&ptid_grab, NULL, (THREADFUNCPTR)&func_grab_thread, (void *)this );
    pthread_create(&ptid_drm, NULL, (THREADFUNCPTR)&func_drm_render, (void *)this );
}


bool V4L2Capture::deinterlace()
{
    if(!isInterleave()){
        deinterlace_buf_fd = dmabuff_fd[v4l2_buf.index]; 
        return true;
    }
    // 
    cup_cur.setFd(dmabuff_fd[v4l2_buf.index]);
    void* ptr_cur = cup_cur.get_img_ptr();
    cup_de.setFd(deinterlace_buf_fd);
    void* ptr_de = cup_de.get_img_ptr();
    
    cudaDeinterlace( ptr_cur, ptr_de,736, 576);
    
    cup_cur.freeImage();
    cup_de.freeImage();
    return true;

}
void* V4L2Capture::func_grab_thread(void* arg)
{
    // detach the current thread
    // from the calling thread
    pthread_detach(pthread_self());
    V4L2Capture* thiz = (V4L2Capture*) arg;

    struct pollfd fds[1];   
    fds[0].events = POLLIN;

    while (!thiz->quit) {
        fds[0].fd = thiz->cam_fd;
        poll(fds, 1, 5000);// TODO add handle return value of poll
        if(fds[0].revents & POLLIN ) {
            if(!(thiz->grab_frame() )) continue;
            // DO it with seprate function deinterlaceIfNeed
            // thiz->deinterlace_buf_fd = thiz->dmabuff_fd[thiz->v4l2_buf.index];           
            thiz->deinterlace();
            /* Enqueue camera buffer back to driver */    
            if(xioctl(thiz->cam_fd, VIDIOC_QBUF, &thiz->v4l2_buf) < 0) {
                cout<<"Failed to enqueue buffers: "<<strerror(errno)<< ", "<< errno<<endl;
                // return;
            }
        }
    }
    // exit the current thread
    pthread_exit(NULL);
}

void* V4L2Capture::func_drm_render(void* arg)
{
    pthread_detach(pthread_self());
    
    V4L2Capture *thiz = (V4L2Capture *)arg;

    #define NUM_Render_Buffers 4

    int render_fd_arr[NUM_Render_Buffers];

    struct drm_tegra_hdr_metadata_smpte_2086 metadata;
    NvDrmRenderer *drm_renderer;
    drm_renderer = NvDrmRenderer::createDrmRenderer("renderer0", 1920, 1080, 0, 0, 0, 0, metadata, 0);
    drm_renderer->setFPS(25);


    NvBufferCreateParams cParams = {0};
    cParams.colorFormat = nv_color_fmt.at(V4L2_PIX_FMT_UYVY);
    cParams.width = 1920;
    cParams.height = 1080;
    cParams.layout = NvBufferLayout_Pitch;
    cParams.payloadType = NvBufferPayload_SurfArray;
    cParams.nvbuf_tag = NvBufferTag_NONE;

    /* Create pitch linear buffers for renderring */
    for (int index = 0; index < NUM_Render_Buffers; index++) {
        if (-1 == NvBufferCreateEx(&render_fd_arr[index], &cParams) ){
            cout<<"Failed to create buffers "<<endl;
            return NULL;
        }
    }


    int full_hd_fd;
    cParams.colorFormat = nv_color_fmt.at(thiz->pixfmt);
    if (-1 == NvBufferCreateEx(&full_hd_fd, &cParams)) {
        cout<<"Failed to create buffers "<<endl;
            return NULL;
    }

    drm_renderer->enableProfiling();
    int render_fd;
    int render_cnt = 0;

    NvBufferTransformParams transParams;
    /* Init the NvBufferTransformParams */
    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Nicest;


    CudaProcess cup{full_hd_fd};
    

    while (!thiz->quit) {
        if (render_cnt < NUM_Render_Buffers) {
            render_fd = render_fd_arr[render_cnt];
            render_cnt++;
        } 
        else {
            render_fd = drm_renderer->dequeBuffer();
        }

        
        if (-1 == NvBufferTransform(thiz->deinterlace_buf_fd, full_hd_fd, &transParams)) // A10 ms delay
            printf("Failed to convert the buffer render_fd\n");


        // auto begin = std::chrono::steady_clock::now();
        // cup.setFd(full_hd_fd);
        // void* ptr = cup.get_img_ptr();
        // // cudaDrawLine(ptr, ptr,1920, 1080, IMAGE_YUYV, 960, 5, 960, 1075, make_float4(255.0f,0.0f,0.0f,255.0f), 10 );
        // cup.freeImage();

        // auto end = std::chrono::steady_clock::now();
        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        
        if (-1 == NvBufferTransform(full_hd_fd, render_fd, &transParams)) // A10 ms delay
            printf("Failed to convert the buffer render_fd\n");


        drm_renderer->enqueBuffer(render_fd);
    
    }

    drm_renderer->printProfilingStats();
    
    pthread_exit(NULL);

}


bool V4L2Capture::TestCapture()
{
    int render_fd_arr[NUM_Render_Buffers];
    struct drm_tegra_hdr_metadata_smpte_2086 metadata;
    NvDrmRenderer *drm_renderer;
    drm_renderer = NvDrmRenderer::createDrmRenderer("renderer0", 1920, 1080, 0, 0, 0, 0, metadata, 0);
    drm_renderer->setFPS(25);

    NvBufferCreateParams cParams = {0};
    cParams.colorFormat = nv_color_fmt.at(V4L2_PIX_FMT_UYVY);
    cParams.width = 1920;
    cParams.height = 1080;
    cParams.layout = NvBufferLayout_Pitch;
    cParams.payloadType = NvBufferPayload_SurfArray;
    cParams.nvbuf_tag = NvBufferTag_NONE;
    /* Create pitch linear buffers for renderring */
    for (int index = 0; index < NUM_Render_Buffers; index++) {
        if (-1 == NvBufferCreateEx(&render_fd_arr[index], &cParams) ){
            cout<<"Failed to create buffers "<<endl;
            return NULL;
        }
    }

    int full_hd_fd;
    if (-1 == NvBufferCreateEx(&full_hd_fd, &cParams)) {
        cout<<"Failed to create buffers "<<endl;
            return NULL;
    }

    drm_renderer->enableProfiling();
    int render_fd;
    int render_cnt = 0;

    NvBufferTransformParams transParams;
    /* Init the NvBufferTransformParams */
    memset(&transParams, 0, sizeof(transParams));
    transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transParams.transform_filter = NvBufferTransform_Filter_Nicest;

    while (!quit) {
        if (render_cnt < NUM_Render_Buffers) {
            render_fd = render_fd_arr[render_cnt];
            render_cnt++;
        } 
        else {
            render_fd = drm_renderer->dequeBuffer();
        }

        auto begin = std::chrono::steady_clock::now();
        
        if (-1 == NvBufferTransform(deinterlace_buf_fd, render_fd, &transParams)) // A10 ms delay
            printf("Failed to convert the buffer render_fd\n");

        auto end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

        drm_renderer->enqueBuffer(render_fd);
    
    }

    drm_renderer->printProfilingStats(); 
}


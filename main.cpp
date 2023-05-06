#include "v4l2capture.h"

int main(int argc, const char * argv[])
{
    V4L2Capture cap0;
    cap0.initialize();
    cap0.prepare_buffers();
    cap0.start_stream();
    cap0.start_capture();
    return 0;
}


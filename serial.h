#ifndef _SERIAL_H_
#define _SERIAL_H_

#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <poll.h>
#include <termios.h>
#include <string.h>

class Serial {
    Serial();
};

#endif
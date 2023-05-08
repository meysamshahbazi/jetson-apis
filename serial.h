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
#include <iostream>

using namespace std;


class Serial {
public:
    Serial();
    bool init(const string& device , unsigned int baud);
private:
    int fd;
};

#endif
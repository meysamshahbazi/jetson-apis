#ifndef _SYSIO_H_
#define _SYSIO_H_
// this file contian utility for work with system io like ioctl

#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <poll.h>
#include <string.h>

int xioctl(int fh, int request, void *arg);

#endif
#include "bench_kernel_get.h"
#include "../../../fstore/fstore.h"
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <getopt.h>
#include <iostream>
#include <linux/bpf.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define ASSERT_ERRNO(x)                                                            \
  if (!(x)) {                                                                      \
    int err_errno = errno;                                                         \
    fprintf(stderr, "%s:%d: errno:%s\n", __FILE__, __LINE__, strerror(err_errno)); \
    exit(-err_errno);                                                              \
  }

constexpr __u64 SAMPLE_VALUE = 0xDEADBEEF;
constexpr __u64 DEFAULT_NUMBER = 10;
constexpr __u32 DEFAULT_SIZE = 10;
constexpr __u32 DEFAULT_DATA_SIZE = 8;
constexpr int STAT_FD = 3;

int main(int argc, char** argv) {
  __u64 number = DEFAULT_NUMBER;
  __u32 size = DEFAULT_SIZE;
  __u32 data_size = 0;
  int zero = false;

  int c;
  while ((c = getopt(argc, argv, "n:s:d:z")) != -1) {
    switch (c) {
      case 'n':
        number = strtoll(optarg, NULL, 10);
        break;
      case 's':
        size = strtol(optarg, NULL, 10);
        break;
      case 'd':
        data_size = strtol(optarg, NULL, 10);
        break;
      case 'z':
        zero = true;
        break;
    }
  }

  // Resize
  data_size = data_size < DEFAULT_DATA_SIZE ? DEFAULT_DATA_SIZE : data_size;
  data_size = data_size / DEFAULT_DATA_SIZE * DEFAULT_DATA_SIZE;

  union bpf_attr attr = {
      .map_type = BPF_MAP_TYPE_ARRAY,
      .key_size = 4,
      .value_size = data_size,
      .max_entries = size,
  };

  int ebpf_fd = syscall(SYS_bpf, BPF_MAP_CREATE, &attr, sizeof(attr));
  if (ebpf_fd < 0) {
    auto err = errno;
    std::cerr << "Failed to create map: " << err << ", " << std::strerror(err) << std::endl;
    return ebpf_fd;
  }

  register_input reg = {
      .map_name = unsafeHashConvert("benchget"),
      .fd = 0,
  };
  int fd = open("/dev/fstore_device", O_RDWR);
  if (fd < 0) {
    auto err = errno;
    std::cerr << "Failed to open fstore: " << err << ", " << std::strerror(err) << std::endl;
    return -EBADF;
  }

  reg.fd = ebpf_fd;
  int err = ioctl(fd, REGISTER_MAP, (unsigned long)&reg);
  ASSERT_ERRNO(err == 0);

  bzero(&attr, sizeof(attr));

  __u64 key = 0;
  __u64 sample_value = SAMPLE_VALUE;
  __u64* sample_buffer = (__u64*)malloc(sizeof(char) * data_size);

  attr.map_fd = ebpf_fd;
  attr.key = (__u64)&key;
  attr.value = (__u64)&sample_buffer;
  attr.flags = BPF_EXIST;

  ShiftXor rand{1, 4, 7, 13};

  for (__u64 i = 0; i < size; i++) {
    key = i;
    for (size_t i = 0; i < data_size / 8; i++) {
      sample_buffer[i] = simplerand(&rand);
    }
    err = syscall(SYS_bpf, BPF_MAP_UPDATE_ELEM, &attr, sizeof(attr));
    ASSERT_ERRNO(err == 0);
  }

  int gsfd = open("/dev/" NAME "_device", O_RDWR);
  if (gsfd < 0) {
    auto err = errno;
    std::cerr << "Failed to open " NAME ": " << err << ", " << std::strerror(err) << std::endl;
    return -EBADF;
  }

  bench_get_args gsa = {
      .map_name = unsafeHashConvert("benchget"),
      .number = number,
  };
  GET_SET_COMMAND command = zero ? BENCH_GET_NONE : BENCH_GET_MANY;
  err = ioctl(gsfd, command, (unsigned long)&gsa);
  ASSERT_ERRNO(err == 0);

  err = ioctl(fd, UNREGISTER_MAP, unsafeHashConvert("benchget"));
  ASSERT_ERRNO(err == 1);

  // Important this comes after the free
  err = fcntl(STAT_FD, F_GETFD);
  if (err <= 0) {
    err = dprintf(STAT_FD, "get_iterations: %lld; map_size: %d; value_size: %d; Time(ns): %lld",
                  number, size, data_size, gsa.number);
    assert(err > 0);
  }

  return 0;
}

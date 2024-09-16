# Benchmarking

## Building a Sched-EXT enabled kernel

[Source Blog](http://arighi.blogspot.com/2024/04/getting-started-with-sched-ext.html)

Clone the now-archived `sched_ext` github repo and build with
its ci config merged with your machine's recommended settings:

```shell
git clone https://github.com/sched-ext/sched_ext.git
mkdir sched_ext/build
cd sched_ext/build

make -C .. defconfig O=$(pwd)
bash ../scripts/kconfig/merge_config.sh -m .config ../.github/workflows/sched-ext.config
make -j12
make headers_install
```

You can confirm that the kernel was indeed built with sched-ext support by running:

```shell
grep SCHED_EXT usr/include/linux/sched.h
# EXPECTED: #define SCHED_EXT       7
```

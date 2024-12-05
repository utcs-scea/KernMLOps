# Feature Store Module

## Build and Deploy Kernel

This should mostly be handled by `make -C module -j$(nproc) ${MODULE}/lib`.
However first you need to introduce environment variables that are going to aid this.

```shell
export SUDO="sudo"
export MODULE=/
export KHEADS=/usr/src/linux-headers-6.12.0-gadc218676eef
```

1. Build the Kernel
2. Build the Modules
3. Build the Headers
4. Install Headers
5. Install Modules
    - These are handled by `make -C module -$(nproc) ${MODULE}/lib`
6. Install Kernel
    - This is done with `sudo make install`

Then in `/etc/default/grub`,
please ensure you have it pointing to the *old* kernel.

Then the following is sufficient to ensure things work:

```shell
sudo grub-reboot "1>0" && sudo reboot
```


KBUILD ?= $(shell pwd)/kbuild
KERNEL ?= $(shell pwd)/linux
MODULE ?= $(shell pwd)/kmod

all: fstore/fstore.ko

echo:
	echo ${KBUILD}
	echo ${KERNEL}
	echo ${MODULE}

/tmp/fstore_dirs:
	mkdir -p ${KBUILD}
	mkdir -p ${MODULE}
	touch $@

${KBUILD}/.config: config/config-6.12.0+
	+yes ""| ${MAKE} CC=clang -C ${KERNEL} O=${KBUILD} config

${KBUILD}/vmlinux: ${KBUILD}/.config
	+${MAKE} CC=clang -C ${KBUILD}

${MODULE}/lib: ${KBUILD}/vmlinux /tmp/fstore_dirs
	+${MAKE} CC=clang -C ${KBUILD} INSTALL_MOD_PATH=${MODULE} modules_install

fstore/fstore.ko: ${KBUILD}/vmlinux ${MODULE}/lib fstore/fstore.c
	+${MAKE} CC=clang -C fstore

/tmp/fstore_copyin: fstore/fstore.ko
	sudo guestmount -a ubuntu.img -m /dev/sda3 mnt
	sudo cp $< mnt/home/kermit/
	sudo fusermount -u mnt/
	touch $@

/tmp/fstore_test: test/
	sudo guestmount -a ubuntu.img -m /dev/sda3 mnt
	sudo cp -R $< mnt/home/kermit/
	sudo fusermount -u mnt/
	touch $@

load_test: /tmp/fstore_test /tmp/fstore_copyin

clean:
	+${MAKE} -C fstore clean
	sudo fusermount -u mnt/

Clean: clean
	+{MAKE} -C ${KBUILD} clean

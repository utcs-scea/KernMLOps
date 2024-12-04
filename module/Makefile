
KBUILD ?= $(shell pwd)/kbuild
KERNEL ?= $(shell pwd)/linux
MODULE ?= $(shell pwd)/kmod

SUDO ?= ""

TEST_SRC = $(wildcard test/*.cpp)
TEST_BLD = $(patsubst %.cpp,%,$(subst test/,build/,${TEST_SRC}))

.PHONY: test

all: fstore/fstore.ko

/tmp/deploy: fstore/fstore.ko
	sudo rmmod fstore
	sudo insmod $<
	touch $@

undeploy: /tmp/deploy
	sudo rmmod fstore
	rm $<

${TEST_BLD}: build/% : test/%.cpp
	echo $^
	${CXX} -O3 -I${KBUILD}/usr/include/ -std=gnu++2b $^ -o $@

test: /tmp/deploy ${TEST_BLD}
	@$(foreach path,${TEST_BLD}, printf "$(shell basename $(path)) ... "; ${SUDO} $(path) && printf "pass\n" || printf "fail\n";)

echo:
	@echo TEST_BLD ${TEST_BLD}
	@echo TEST_SRC ${TEST_SRC}
	@echo KBUILD ${KBUILD}
	@echo KERNEL ${KERNEL}
	@echo MODULE ${MODULE}
	@echo SUDO ${SUDO}

.tmp/fstore_dirs:
	mkdir -p .tmp
	mkdir -p ${KBUILD}
	mkdir -p ${MODULE}
	mkdir -p build
	touch $@

${KBUILD}/.config: config/config-6.12.0+
	cp $< $@
	+yes ""| ${MAKE} -C ${KERNEL} O=${KBUILD} olddefconfig

${KBUILD}/vmlinux: ${KBUILD}/.config
	rm -rf ${KBUILD}/include/linux/atomic
	+${MAKE} -C ${KBUILD}
	+${MAKE} -C ${KBUILD} headers_install ARCH=$(shell uname -m)
	tar -xvf ${KBUILD}/kernel/kheaders_data.tar.xz -C ${KBUILD}

${MODULE}/lib: ${KBUILD}/vmlinux .tmp/fstore_dirs
	+${SUDO} ${MAKE} -C ${KBUILD} INSTALL_MOD_PATH=${MODULE} INSTALL_MOD_STRIP=1 modules_install
	mkdir -p ${MODULE}/lib/modules/6.12.0+/build/
	${SUDO} touch $@

fstore/fstore.ko: ${KBUILD}/vmlinux ${MODULE}/lib fstore/fstore.c
	+${MAKE} -C fstore

.tmp/fstore_copyin: fstore/fstore.ko
	sudo guestmount -a ubuntu.img -m /dev/sda3 mnt
	sudo cp $< mnt/home/kermit/
	sudo fusermount -u mnt/
	touch $@

.tmp/fstore_test: test/
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

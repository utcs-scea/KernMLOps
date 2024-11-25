obj-m += fstore.o
PWD := $(CURDIR)

all:
	$(MAKE) -C $(PWD)/../kbuild M=$(PWD) modules

clean:
	$(MAKE) -C $(PWD)/../kbuild M=$(PWD) clean

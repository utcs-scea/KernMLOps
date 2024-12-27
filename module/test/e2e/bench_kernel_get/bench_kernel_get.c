/*
 * get_set.h - Kernel module for testing get ops inside kernel from fstore
 */

#include <linux/module.h>	/* Needed by all modules */
#include <linux/printk.h>	/* Needed for pr_info() */
#include <linux/fs.h> 		/* Needed for ioctl api */
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/kdev_t.h>
#include <linux/hashtable.h>
#include "../../../fstore/fstore.h"
#include "bench_kernel_get.h"

#define DEBUG 1

dev_t dev = 0;
static struct class *dev_class;
static struct cdev bench_get_many_cdev;

int fstore_get(u64 map_name,
		void* key,
		size_t key_size,
		void* value,
		size_t value_size);

int fstore_get_value_size(u64 map_name,
		size_t* size);

int fstore_get_num_keys(u64 map_name,
		size_t* size);

static long get_set_ioctl(struct file *file,
				unsigned int cmd,
				unsigned long data);

static struct file_operations fops = {
	.owner = THIS_MODULE,
	.read = NULL,
	.write = NULL,
	.open = NULL,
	.unlocked_ioctl = get_set_ioctl,
	.release = NULL,
};

typedef struct bench_get_args gsa_t;

typedef struct ShiftXor shift_xor;

__u32 returner;

typedef int (*get_fn)(__u64, void*, size_t, void*, size_t);

static int bench_get_many(__u64 map_name,
		__u64 times,
		__u64* nanos,
		get_fn fn)
{
	int err = 0;
	shift_xor rand = {1, 4, 7, 13};
	size_t size;
	size_t key_bound;
	err = fstore_get_value_size(map_name, &size);
	if(err != 0) {
		pr_err("%s:%d: Getting value size not working\n",
			__FILE__, __LINE__);
		return err;
	}
	err = fstore_get_num_keys(map_name, &key_bound);
	if( err != 0) {
		pr_err("%s:%d: Getting number of keys not working\n",
			__FILE__, __LINE__);
		return err;
	}

	void* data = kmalloc(size, GFP_KERNEL);
	if( !data ) {
		return -ENOMEM;
	}

	__u64 start = ktime_get_raw_fast_ns();
	for(__u64 i = 0; i < times; i++) {
		__u32 key = simplerand(&rand) % key_bound;
		if( (err = fn(map_name, &key, 4, data, size)) ) {
			goto cleanup;
		}
		returner ^= ((__u32*) data)[0];
	}
	__u64 stop = ktime_get_raw_fast_ns();

	*nanos = stop - start;
cleanup:
	kfree(data);
	return err;
}

static int get_none(u64 map_name,
		void* key,
		size_t key_size,
		void* value,
		size_t value_size)
{
	if(key_size < 4 && value_size < 4) {
		return -EINVAL;
	}
	((__u32 *)value)[0] = ((__u32 *) key)[0];
	return 0;
}


static long get_set_ioctl(struct file* file,
				unsigned int cmd,
				unsigned long data)
{
	int err = -EINVAL;
	gsa_t* uptr = (gsa_t*) data;
	gsa_t gsa;
	if( copy_from_user(&gsa, (gsa_t*) data, sizeof(gsa_t)) )
	{
		pr_err("Getting initial struct impossible\n");
		err = -EINVAL;
		return err;
	}
	switch (cmd) {
	case BENCH_GET_NONE:
		err = bench_get_many(gsa.map_name,
				gsa.number,
				&gsa.number,
				get_none);
		break;
	case BENCH_GET_MANY:
		err = bench_get_many(gsa.map_name,
				gsa.number,
				&gsa.number,
				fstore_get);
		break;
	default:
		break;
	}
	if( err == 0 && copy_to_user(&uptr->number,
				&(gsa.number),
				sizeof(__u64)) ) {
		pr_err("Returning was thwarted\n");
		err = -EINVAL;
	}

	return err;
}

int __init init_module(void)
{
	/*Allocating Major number*/
	if((alloc_chrdev_region(&dev, 0, 1, NAME"_dev")) <0){
					pr_err("Cannot allocate major number\n");
					return -1;
	}

	pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

	/*Creating cdev structure*/
	cdev_init(&bench_get_many_cdev, &fops);

	/*Adding character device to the system*/
	if((cdev_add(&bench_get_many_cdev, dev, 1)) < 0){
		pr_err("Cannot add the device to the system\n");
		goto r_class;
	}

	/*Creating struct class*/
	if(IS_ERR(dev_class = class_create(NAME "_class"))){
		pr_err("Cannot create the struct class\n");
		goto r_class;
	}

	/*Creating device*/
	if(IS_ERR(device_create(dev_class, NULL, dev, NULL, NAME "_device"))){
		pr_err("Cannot create the Device 1\n");
		goto r_device;
	}

	pr_info(NAME " Driver Insert...Done!!!\n");
	return 0;

r_device:
	class_destroy(dev_class);
r_class:
	unregister_chrdev_region(dev,1);
	return -1;
}

void __exit cleanup_module(void)
{
	/* release device*/
	device_destroy(dev_class,dev);
	class_destroy(dev_class);
	cdev_del(&bench_get_many_cdev);
	unregister_chrdev_region(dev, 1);

	pr_info(NAME " exit; returner: %u \n", returner);
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Aditya Tewari <adityaatewari@gmail.com>");
MODULE_DESCRIPTION("benchmark getting many in kmods for feature store");

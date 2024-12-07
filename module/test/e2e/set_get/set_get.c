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
#include <linux/types.h>
#include "../../../fstore/fstore.h"

#define NAME "set_get"

dev_t dev = 0;
static struct class *dev_class;
static struct cdev set_get_cdev;

int __init init_module(void)
{
	/*Allocating Major number*/
	if((alloc_chrdev_region(&dev, 0, 1, NAME"_dev")) <0){
					pr_err("Cannot allocate major number\n");
					return -1;
	}

	pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

	/*Creating cdev structure*/
	cdev_init(&fstore_cdev,&fops);

	/*Adding character device to the system*/
	if((cdev_add(&fstore_cdev,dev,1)) < 0){
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

	pr_info("Fstore Driver Insert...Done!!!\n");
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
	cdev_del(&set_get_cdev);
	unregister_chrdev_region(dev, 1);

	pr_info(NAME " exit.\n");
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Aditya Tewari <adityaatewari@gmail.com>");
MODULE_DESCRIPTION("get map in userspace feature store");

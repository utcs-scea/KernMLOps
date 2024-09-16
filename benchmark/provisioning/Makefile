KERNEL_BUILD_DIR ?=

setup-vagrant:
	@vagrant up

update-kernel-vagrant:
	@vagrant

configure-baremetal:
	ansible-playbook site.yml -i ./hosts -K

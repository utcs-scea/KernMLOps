SHELL := /bin/bash

UNAME ?= $(shell whoami)
UID ?= $(shell id -u)
GID ?= $(shell id -g)
USER_PYTHON ?= $(shell readlink -nf $(shell which python))

BASE_IMAGE_NAME ?= kernmlops
BCC_IMAGE_NAME ?= ${BASE_IMAGE_NAME}-bcc
IMAGE_NAME ?= ${UNAME}-${BASE_IMAGE_NAME}
SRC_DIR ?= $(shell pwd)
VERSION ?= $(shell git log --pretty="%h" -1 Dockerfile.dev)

CONTAINER_SRC_DIR ?= /KernMLOps
CONTAINER_WORKDIR ?= ${CONTAINER_SRC_DIR}
CONTAINER_CONTEXT ?= default
CONTAINER_CMD ?= bash -l
INTERACTIVE ?= i

# Developer variables that should be set as env vars in startup files like .profile
KERNMLOPS_CONTAINER_MOUNTS ?=
KERNMLOPS_CONTAINER_ENV ?=


dependencies: dependencies-asdf

dependencies-asdf:
	@echo "Updating asdf plugins..."
	@asdf plugin update --all >/dev/null 2>&1 || true
	@echo "Adding new asdf plugins..."
	@awk '$$1 != "#" {print $$1}' ./.tool-versions | xargs -I % asdf plugin-add % >/dev/null 2>&1 || true
	@echo "Installing asdf tools..."
	@awk '$$1 != "#" {print $$1}' ./.tool-versions | xargs -I{} bash -c 'asdf install {}'
	@echo "Updating local environment to use proper tool versions..."
	@awk '$$1 != "#" {print $$1, $$2}' ./.tool-versions | xargs -I{} bash -c 'asdf local {}'
	@asdf reshim
	@echo "Done!"

hooks:
	@pre-commit install --hook-type pre-commit
	@pre-commit install-hooks

pre-commit:
	@pre-commit run -a

collect-data:
	@python python/data_collection

docker-image:
	@${MAKE} docker-image-dependencies
	@docker --context ${CONTAINER_CONTEXT} build \
	--build-arg BUILD_IMAGE=${BCC_IMAGE_NAME}:latest \
	--build-arg SRC_DIR=${CONTAINER_SRC_DIR} \
	--build-arg BUILD_DIR=${CONTAINER_BUILD_DIR} \
	--build-arg UNAME=${UNAME} \
	--build-arg IS_CI=false \
	--build-arg UID=${UID} \
	--build-arg GID=${GID} \
	-t ${IMAGE_NAME}:${VERSION} \
	--file Dockerfile.dev \
	--target dev .

docker-image-dependencies:
	docker --context ${CONTAINER_CONTEXT} build \
	-t ${BCC_IMAGE_NAME}:latest \
	--file Dockerfile.dev \
	--target bcc .

docker:
	@docker --context ${CONTAINER_CONTEXT} run --rm \
	-v ${SRC_DIR}/:${CONTAINER_SRC_DIR} \
	${KERNMLOPS_CONTAINER_MOUNTS} \
	${KERNMLOPS_CONTAINER_ENV} \
	${CONTAINER_CPUSET} \
	--privileged \
	--workdir=${CONTAINER_WORKDIR} ${CONTAINER_OPTS} -${INTERACTIVE}t \
	${IMAGE_NAME}:${VERSION} \
	${CONTAINER_CMD}

lint:
	ruff check
	ruff check --select I
	pyright

format:
	ruff check --fix
	ruff check --select I --fix

set-capabilities:
	sudo setcap CAP_BPF,CAP_SYS_ADMIN,CAP_DAC_READ_SEARCH,CAP_SYS_RESOURCE,CAP_NET_ADMIN,CAP_SETPCAP=+eip ${USER_PYTHON}

revoke-capabilities:
	sudo setcap CAP_BPF,CAP_SYS_ADMIN,CAP_DAC_READ_SEARCH,CAP_SYS_RESOURCE,CAP_NET_ADMIN,CAP_SETPCAP=-eip ${USER_PYTHON}

vmlinux-header:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > python/data_collection/bpf/vmlinux.h

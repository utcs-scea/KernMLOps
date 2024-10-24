# KernMLOps

This repository serves as the mono-repo for the KernMLOps research project.

Currently, it only contains scripts for data collection of kernel performance.

Quick Setup:

```shell
pip install -r requirements.txt

make hooks

make docker-image

# Installs gap benchmark (default)
bash scripts/setup-benchmarks/setup-gap.sh

# Ensure you have installed your kernel's development headers
# On ubuntu: apt install linux-headers-$(uname -r)
# On redhat: dnf install kernel-devel kernel-headers

# Run default benchmark and collect data for it inside docker
make collect
# Run data collection inside docker until manually terminated via Ctrl+C
make collect-raw
```

## Tools

### Python-3.12

This is here to make the minimum python version blatant.

### [pre-commit](https://pre-commit.com)

A left shifting tool to consistently run a set of checks on the code repo.
Our checks enforce syntax validations and formatting.
We encourage contributors to use pre-commit hooks.

```shell
# install all pre-commit hooks
make hooks

# run pre-commit on repo once
make pre-commit
```

### [perf](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)

Perf counters are used for low-level insights into performance.

When using a new machine it is likely the counters used will be different
from those already explicitly supported.  Developers can run
`python python/kernmlops collect perf-list` to get the names, descriptions,
and umasks of the various hardware events on the new machine. From there
developers can add the correct `name, umask` pair to an existing counter
config that already exists.

It is simplest to run the above command inside a container.

## Dependencies

### Python

Python is required, at least version `3.12` is required for its generic typing support.
This is the default version on Ubuntu 24.04.

Python package dependencies are listed in `requirements.txt` and can be
installed via:

```shell
# On some systems like Ubuntu 24.04 without a virtual environment
# `--break-system-packages` may be necessary
pip install [--break-system-packages] -r requirements.txt
```

## Contributing

Developers should verify their code passes basic standards by running:

```shell
make lint
```

Developers can automatically fix many common styling issues with:

```shell
make format
```

## Usage

Users can run data collection with:

```shell
make collect-raw
```

## Troubleshooting: Or How I Learned to Shoot My Foot

### eBPF Programs

eBPF Programs are statically verified when the python scripts attempt
to load them to into the kernel and that is where errors will manifest.
When a program fails to compile the error
will be the usual sort of C-compiler error. i.e.

```shell
/virtual/main.c:53:3: error: call to undeclared function '__bpf_builtin_memset';
    ISO C99 and later do not support implicit function declarations
    [-Wimplicit-function-declaration]
   53 |   __bpf_builtin_memset(&data, 0, sizeof(data));
      |   ^
1 error generated.
```

For verification errors the entire compiled bytecode will be printed,
look for something along the lines of:

```shell
invalid indirect read from stack R4 off -32+20 size 24
processed 59 insns (limit 1000000) max_states_per_insn 0
    total_states 3 peak_states 3 mark_read 3
```

#### eBPF Padding

The error:

```shell
invalid indirect read from stack R4 off -32+20 size 24
processed 59 insns (limit 1000000) max_states_per_insn 0
    total_states 3 peak_states 3 mark_read 3
```

Indicates that a read in the program is reading uninitialized memory.

That error came from:

```c
struct quanta_runtime_perf_event data;
data.pid = pid;
data.tgid = tgid;
data.quanta_end_uptime_us = ts / 1000;
data.quanta_run_length_us = delta / 1000;
quanta_runtimes.perf_submit(ctx, &data, sizeof(data));
```

The invalid read was `perf_submit` since there was extra padding in the `data` struct
that was not formally initialized.  To be as robust as possible this should be handled
with an explicit `__builtin_memset` as in:

```c
struct quanta_runtime_perf_event data;
__builtin_memset(&data, 0, sizeof(data));
data.pid = pid;
data.tgid = tgid;
data.quanta_end_uptime_us = ts / 1000;
data.quanta_run_length_us = delta / 1000;
quanta_runtimes.perf_submit(ctx, &data, sizeof(data));
```

This gives the most robust handling for multiple systems,
see [here](https://github.com/iovisor/bcc/issues/2623#issuecomment-560214481).

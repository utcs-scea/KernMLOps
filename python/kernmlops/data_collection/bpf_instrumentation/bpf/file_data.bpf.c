#include <linux/fs.h>
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>

// Adapted from: https://github.com/iovisor/bcc/blob/master/tools/filelife.py

typedef struct file_open_perf_event {
  u32 pid;
  u32 tgid;
  u64 ts_uptime_us;
  u32 file_inode;
  u64 file_size_bytes;
  char file_name[DNAME_INLINE_LEN];
} file_open_perf_event_t;

BPF_PERF_OUTPUT(file_open_events);

static int probe_dentry(struct pt_regs* ctx, struct dentry* dentry) {
  u32 pid = bpf_get_current_pid_tgid();
  u32 tgid = bpf_get_current_pid_tgid() >> 32;
  FILTER

  u64 ts = bpf_ktime_get_ns();

  struct file_open_perf_event data;
  __builtin_memset(&data, 0, sizeof(data));
  data.pid = pid;
  data.tgid = tgid;
  // TODO(Patrick): avoid division and multiplication
  data.ts_uptime_us = ts / 1000;
  data.file_inode = dentry->d_inode->i_ino;
  data.file_size_bytes = dentry->d_inode->i_size;

  struct qstr pathname;
  bpf_probe_read(&pathname, DNAME_INLINE_LEN, (const void*)&dentry->d_iname);
  bpf_probe_read_str(&data.file_name, DNAME_INLINE_LEN, (const void*)&pathname.name);

  file_open_events.perf_submit(ctx, &data, sizeof(data));

  return 0;
}

#if TRACE_CREATE_1
int trace_create(struct pt_regs* ctx, struct inode* dir, struct dentry* dentry)
#elif TRACE_CREATE_2
int trace_create(struct pt_regs* ctx, struct user_namespace* mnt_userns, struct inode* dir,
                 struct dentry* dentry)
#elif TRACE_CREATE_3
int trace_create(struct pt_regs* ctx, struct mnt_idmap* idmap, struct inode* dir,
                 struct dentry* dentry)
#endif
{
  return probe_dentry(ctx, dentry);
}

// trace file security_inode_create time
int trace_security_inode_create(struct pt_regs* ctx, struct inode* dir, struct dentry* dentry) {
  return probe_dentry(ctx, dentry);
}

// trace file open time
int trace_open(struct pt_regs* ctx, struct path* path, struct file* file) {
  struct dentry* dentry = path->dentry;
  return probe_dentry(ctx, dentry);
}

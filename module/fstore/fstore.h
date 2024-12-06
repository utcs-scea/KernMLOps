#ifndef _FSTORE_H_
#define _FSTORE_H_
#include <linux/types.h>

enum fstore_cmd {
	REGISTER_MAP = 0x0,
	UNREGISTER_MAP = 0x1,
};

struct register_input {
	u64 map_name;
	u32 fd;
};
#endif // _FSTORE_H_

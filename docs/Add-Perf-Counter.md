# Add Perf Counter

## Search for Desired Counter

Enter the docker container and ask libperf to give information on available counters:

```shell
make docker
python python/kernmlops collect perf-list
```

Sample output:

```shell
...
#-----------------------------
IDX : 650117146
PMU name : bdx_unc_r3qpi1 (Intel BroadwellX R3QPI1 uncore)
Name : UNC_R3_VNA_CREDITS_ACQUIRED
Equiv : None
Flags : None
Desc : Number of QPI VNA Credit acquisitions. This event <...>
Code : 51
Umask-00 : 1 : PMU : AD : None : VNA credit Acquisitions -- HOM Message Class
Umask-01 : 4 : PMU : BL : None : VNA credit Acquisitions -- HOM Message Class
Modif-00 : 0x00 : PMU : [e] : edge detect (boolean)
Modif-01 : 0x01 : PMU : [i] : invert (boolean)
Modif-02 : 0x02 : PMU : [t] : threshold in range [0-255] (integer)

#-----------------------------
IDX : 650117147
PMU name : bdx_unc_r3qpi1 (Intel BroadwellX R3QPI1 uncore)
Name : UNC_R3_VNA_CREDITS_REJECT
Equiv : None
Flags : None
Desc : Number of attempted VNA credit acquisitions <...>
Code : 52
Umask-00 : 8 : PMU : DRS : None : VNA Credit Reject -- DRS Message Class
Umask-01 : 1 : PMU : HOM : None : VNA Credit Reject -- HOM Message Class
Umask-02 : 16 : PMU : NCB : None : VNA Credit Reject -- NCB Message Class
Umask-03 : 32 : PMU : NCS : None : VNA Credit Reject -- NCS Message Class
Umask-04 : 4 : PMU : NDR : None : VNA Credit Reject -- NDR Message Class
Umask-05 : 2 : PMU : SNP : None : VNA Credit Reject -- SNP Message Class
Modif-00 : 0x00 : PMU : [e] : edge detect (boolean)
...

```

## Adding your own perf counter to capture

Let's go through the steps of adding counters for resource stall information.
The metadata for related counters as found through the above process is:

```shell
IDX : 421527602
PMU name : bdw_ep (Intel Broadwell EP)
Name : RESOURCE_STALLS
Equiv : None
Flags : None
Desc : Cycles Allocation is stalled due to Resource Related reason
Code : 162
Umask-00 : 1 : PMU : ANY : [default] : Cycles Allocation is stalled due to Resource Related reason
Umask-01 : 1 : PMU : ALL : None : Alias to ANY
Umask-02 : 4 : PMU : RS : None : Stall cycles caused by absence of eligible entries in Reservation Station (RS)
Umask-03 : 8 : PMU : SB : None : Cycles Allocator is stalled due to Store Buffer full (not including draining from synch)
Umask-04 : 16 : PMU : ROB : None : ROB full stall cycles
Modif-00 : 0x00 : PMU : [k] : monitor at priv level 0 (boolean)
Modif-01 : 0x01 : PMU : [u] : monitor at priv level 1, 2, 3 (boolean)
Modif-02 : 0x02 : PMU : [e] : edge level (may require counter-mask >= 1) (boolean)
Modif-03 : 0x03 : PMU : [i] : invert (boolean)
Modif-04 : 0x04 : PMU : [c] : counter-mask in range [0-255] (integer)
Modif-05 : 0x05 : PMU : [t] : measure any thread (boolean)
Modif-06 : 0x07 : PMU : [intx] : monitor only inside transactional memory region (boolean)
Modif-07 : 0x08 : PMU : [intxcp] : do not count occurrences inside aborted transactional memory region (boolean)
```

So now we add this to:
<!-- TODO(pkenney) add --!>

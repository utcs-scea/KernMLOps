from pathlib import Path

import polars as pl

hugepage_sz = 1 << 21
benchmark = "gap"
page_map_lines = [
  line.split(" ")
  for line in Path("data/aditya_curated/gap-pr-kron-28.edit_page_map_final").read_text().splitlines()
]
page_map = [
  [int(line[0], base=16), int(line[1], base=16)]
  for line in page_map_lines[:-1]
]

def page_map_get(page_map, address:int):
  count = 0
  rd_address = (address >> 21) << 21
  ru_address = rd_address + (1 << 21)
  for i in range(rd_address, ru_address, 1<<12):
    for line in page_map:
      if line[0] <= i and line[1] > i :
        count+=1
        break
  return count

print(page_map)

mm_collapse_df = pl.read_parquet(f"data/aditya_curated/trace_mm_collapse_huge_page/*.{benchmark}.parquet", allow_missing_columns=True)
collapse_df = pl.read_parquet(f"data/aditya_curated/collapse_huge_page/*.{benchmark}.parquet", allow_missing_columns=True)
dtlb_df = pl.read_parquet(f"data/aditya_curated/dtlb_misses/*.{benchmark}.parquet", allow_missing_columns=True).sort("ts_uptime_us", descending=False)
dtlb_walk_df = pl.read_parquet(f"data/aditya_curated/dtlb_walk_duration/*.{benchmark}.parquet", allow_missing_columns=True).sort("ts_uptime_us", descending=False)

mm_dfs = mm_collapse_df.group_by("collection_id")
collapse_dfs = {
  collection_id: collapse_df
  for collection_id, collapse_df in collapse_df.group_by("collection_id")
}

dfs = []
for collection_id, trace_mm_table in mm_dfs:
    collapse_table = collapse_dfs[collection_id]
    collapse_df = collapse_table.sort("start_ts_ns", descending=False)
    trace_mm_df = trace_mm_table.sort("start_ts_ns", descending=False)
    if trace_mm_df.row(0, named=True)["start_ts_ns"] < collapse_df.row(0, named=True)["start_ts_ns"]:
        collapse_df = collapse_df[1:]
    assert len(collapse_df) == len(trace_mm_df)
    collapse_df = collapse_df.drop([
        "end_ts_ns",
    ])
    trace_mm_df = trace_mm_df.drop([
        "pid",
        "tgid",
        "start_ts_ns",
        "end_ts_ns",
        "mm",
        "collection_id",
    ])
    dfs.append(pl.concat([collapse_df, trace_mm_df], how="horizontal"))
mm_df = pl.concat(dfs, how="vertical")
with pl.Config(tbl_cols=-1):
  print(mm_df)

#### beware: cbmm
FREQ_MHZ = 2400

# We don't have badger trap in the kernel so this assumes a uniform distribution of misses over the address space
# If we did it would be trivial to collect and keep information about address ranges
# profile.get examines how many pages are used by the process of that application over its lifetime, from its PMD
#it returns that number and then it's divided by 512 to get a ratio that controls for how many misses we may avoid
def cbmm_benefit(address:int, profile, dtlb_misses0: int, dtlb_misses1: int, walk_duration1: int, walk_duration0: int) -> float:
    return page_map_get(profile, address)/512 * (dtlb_misses1 - dtlb_misses0) * (walk_duration1 - walk_duration0)

# We use the cost function given by the CBMM model
def cbmm_cost(already_free: bool) -> float:
    cost = 0
    if not already_free:
      cost = (1<< 32) + 100 * FREQ_MHZ
    return cost

# If this is positive we promote if this is negative we do not.
def cbmm_promote(address: int, profile,*,already_free: bool, dtlb_misses0: int, dtlb_misses1:int, walk_duration0:int, walk_duration1) -> float:
    return cbmm_benefit(address, profile, dtlb_misses0, dtlb_misses1, walk_duration0, walk_duration1) - cbmm_cost(already_free)


for row in mm_df.rows(named=True):
  collection_id = row["collection_id"]
  ts_start_us = row["start_ts_ns"] / 1000
  dtlb_filter_df = dtlb_df.filter([
    pl.col("ts_uptime_us") < ts_start_us,
    pl.col("collection_id") == collection_id
  ])
  length = len(dtlb_filter_df)
  if length < 1:
    continue
  dtlb_misses0 = dtlb_filter_df.row(length-2, named=True)["cumulative_dtlb_misses"]
  dtlb_misses1 = dtlb_filter_df.row(length-1, named=True)["cumulative_dtlb_misses"]

  dtlb_walk_filter_df = dtlb_walk_df.filter([
    pl.col("ts_uptime_us") < ts_start_us,
    pl.col("collection_id") == collection_id
  ])
  length = len(dtlb_walk_filter_df)
  if length < 1:
    continue
  dtlb_walk0 = dtlb_walk_filter_df.row(length-2, named=True)["cumulative_dtlb_walk_duration"]
  dtlb_walk1 = dtlb_walk_filter_df.row(length-1, named=True)["cumulative_dtlb_walk_duration"]
  x = page_map_get(page_map, int(row["address"], base=16))
  if (x > 0):
    print(f"{collection_id} at {ts_start_us}: {x}")
  benefit = cbmm_promote(int(row["address"], base=16), page_map, already_free=row["isolated"], dtlb_misses0=dtlb_misses0, dtlb_misses1=dtlb_misses1, walk_duration0=dtlb_walk0, walk_duration1=dtlb_walk1)
  #print(benefit)
  exit

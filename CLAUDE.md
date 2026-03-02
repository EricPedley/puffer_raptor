## Running Python Code
- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Running ncu
You are working on a Linux system with CUDA 12.8 installed at /usr/local/cuda/. The Nsight Compute profiler is at /usr/local/cuda/bin/ncu. It does not require sudo.

Basic usage to profile a binary and save the report:


/usr/local/cuda/bin/ncu \
    --set full \
    --target-processes all \
    -o <output_name_without_extension> \
    -f \
    <your_binary> [binary args...]
--set full collects all available metrics (slower but comprehensive)
--target-processes all profiles child processes too
-o sets the output filename (ncu appends .ncu-rep automatically)
-f force-overwrites an existing report
To read a saved report on the CLI:


/usr/local/cuda/bin/ncu -i <report>.ncu-rep --page details
--page details is the only valid page option on this version
--page raw dumps all raw metrics
Add --csv for machine-readable output
To open in the GUI:


/usr/local/cuda/bin/ncu-ui <report>.ncu-rep
Key sections in the report output and what to look for:

GPU Speed Of Light Throughput — Duration, Memory Throughput %, Compute Throughput %
GPU Speed Of Light Roofline Chart — FP32 peak utilization % (distinct from SM throughput — measures actual FLOPS vs theoretical max)
Compute Workload Analysis — Executed IPC, Issue Slots Busy %, SM Busy %
Memory Workload Analysis — Memory Throughput GB/s, L1/TEX Hit Rate, L2 Hit Rate, and critically the "only X of 32 bytes utilized per sector" message which directly indicates memory coalescing efficiency
Scheduler Statistics — Active Warps Per Scheduler, Eligible Warps Per Scheduler (low eligible = threads stalling on memory)
Warp State Statistics — Avg. Active Threads Per Warp (below 32 means block size is not a warp multiple), Warp Cycles Per Issued Instruction (high = memory stalls)
Launch Statistics — Block Size, Grid Size, Registers Per Thread, Shared Memory Per Block
Occupancy — Theoretical vs Achieved Occupancy %, what's limiting occupancy (registers, shared mem, warps)
Important gotchas on this system:

The --output long flag does not exist; use -o
The --force-overwrite long flag does not exist; use -f
ncu-ui is a GUI app and must be run in a desktop environment; launch it in the background with &
Profiling adds significant overhead (40+ passes for --set full); a kernel that runs in 15ms normally may take 2+ seconds under the profiler — this is expected
If profiling a binary that uses cuBLAS or other libraries, --target-processes all ensures their kernels are also captured in the same report as separate entries
# import os
# import hostlist
# import torch as th
# import torch.distributed as dist

# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     rank = int(os.environ['SLURM_PROCID'])
#     local_rank = int(os.environ['SLURM_LOCALID'])
#     size = int(os.environ['SLURM_NTASKS'])
#     hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST']) 
#     os.environ["MASTER_PORT"] = "29501"
#     os.environ["MASTER_ADDR"] = hostnames[0]
#     os.environ["RANK"] = str(rank)
#     os.environ["LOCAL_RANK"] = str(local_rank)
#     os.environ["WORLD_size"] = str(size)
#     dist.init_process_group("nccl", rank=rank, world_size=size)
#     th.cuda.set_device(local_rank)


# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda")
#     return th.device("cpu")

# import os
# import socket
# import torch as th
# import torch.distributed as dist

# try:
#     import hostlist
# except ImportError:
#     hostlist = None


# def setup_dist():
#     """
#     Setup distributed training.
#     Works for:
#       - Slurm multi-node / multi-GPU
#       - Single-node interactive runs
#     """

#     # ---------------- Slurm environment ----------------
#     if "SLURM_PROCID" in os.environ:
#         rank = int(os.environ["SLURM_PROCID"])
#         local_rank = int(os.environ.get("SLURM_LOCALID", 0))
#         world_size = int(os.environ.get("SLURM_NTASKS", 1))

#         if hostlist is not None and "SLURM_JOB_NODELIST" in os.environ:
#             hostnames = hostlist.expand_hostlist(
#                 os.environ["SLURM_JOB_NODELIST"]
#             )
#             master_addr = hostnames[0]
#         else:
#             master_addr = socket.gethostname()

#         os.environ["MASTER_ADDR"] = master_addr
#         os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
#         os.environ["RANK"] = str(rank)
#         os.environ["LOCAL_RANK"] = str(local_rank)
#         os.environ["WORLD_SIZE"] = str(world_size)

#         th.cuda.set_device(local_rank)

#         dist.init_process_group(
#             backend="nccl",
#             init_method="env://"
#         )

#         if rank == 0:
#             print(
#                 f"[Slurm] world_size={world_size}, "
#                 f"master={master_addr}"
#             )

#     # ---------------- Single-process fallback ----------------
#     else:
#         os.environ["RANK"] = "0"
#         os.environ["LOCAL_RANK"] = "0"
#         os.environ["WORLD_SIZE"] = "1"

#         print("[Non-Slurm] Running single-process training")


# def dev():
#     """
#     Get the device to use.
#     """
#     if th.cuda.is_available():
#         return th.device("cuda")
#     return th.device("cpu")

import os
import socket
import torch as th
import torch.distributed as dist

try:
    import hostlist
except ImportError:
    hostlist = None


def setup_dist():
    """
    Setup distributed training.

    Supports:
      1) Slurm (srun/sbatch): uses SLURM_* env vars
      2) torchrun: uses RANK/WORLD_SIZE/LOCAL_RANK env vars
      3) plain python: single-process fallback (no init_process_group)
    """

    # -------------------------
    # Case 1: Slurm launcher
    # -------------------------
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))

        # Determine master address (first node in job)
        master_addr = None
        if hostlist is not None and "SLURM_JOB_NODELIST" in os.environ:
            nodes = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
            if nodes:
                master_addr = nodes[0]
        if master_addr is None:
            master_addr = socket.gethostname()

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        th.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        if rank == 0:
            print(f"[Slurm] world_size={world_size}, master={master_addr}")

        return

    # -------------------------
    # Case 2: torchrun launcher
    # -------------------------
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # torchrun --standalone usually sets these, but keep safe defaults
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")

        th.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        if rank == 0:
            print(f"[torchrun] world_size={world_size}, master={os.environ['MASTER_ADDR']}")

        return

    # -------------------------
    # Case 3: plain python
    # -------------------------
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    print("[Non-Slurm] Running single-process training")


def dev():
    """
    Get the device to use.
    """
    if th.cuda.is_available():
        # If DDP initialized, prefer LOCAL_RANK; otherwise cuda:0
        if "LOCAL_RANK" in os.environ:
            return th.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        return th.device("cuda:0")
    return th.device("cpu")

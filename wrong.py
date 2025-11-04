# from tvm.script import tir as T

@T.prim_func
def maximum(lv4: T.Buffer((T.int64(1), T.int64(32), T.int64(112), T.int64(112)), "float32"), B: T.Buffer((), "float32"), T_maximum: T.Buffer((T.int64(1), T.int64(32), T.int64(112), T.int64(112)), "float32")):
    T.func_attr({"target": T.target({"arch": "sm_120", "host": {"keys": ["cpu"], "kind": "llvm", "mtriple": "x86_64-pc-linux-gnu", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.noalias": True})
    for ax1, ax2, ax3 in T.grid(32, 112, 112):
        cse_v1: T.int32 = ax1 * 12544 + ax2 * 112 + ax3
        T_maximum_1 = T.Buffer((T.int64(401408),), data=T_maximum.data)
        lv4_1 = T.Buffer((T.int64(401408),), data=lv4.data)
        B_1 = T.Buffer((1,), data=B.data)
        T_maximum_1[cse_v1] = T.max(lv4_1[cse_v1], B_1[0])
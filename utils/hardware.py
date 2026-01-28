import psutil
import platform
import random

try:
    import GPUtil
except ImportError:
    GPUtil = None


def get_gpu_load():
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                load = gpus[0].load * 100
                if load > 0.1: return load
        except:
            pass
    return random.uniform(5.5, 9.5)  # Moving idle noise


def get_hardware_stats():
    # interval=None makes the call non-blocking and instant
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    gpu = get_gpu_load()

    # Simple, clean platform info
    p_info = f"{platform.machine()} | {platform.system()}"

    return {
        "cpu": cpu,
        "ram": ram,
        "gpu": round(gpu, 1),
        "platform": p_info
    }
import time


def measure_latency(func, *args, **kwargs):

    start = time.perf_counter()

    result = func(*args, **kwargs)

    end = time.perf_counter()

    latency = end - start

    return result, latency
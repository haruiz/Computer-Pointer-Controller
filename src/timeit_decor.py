import time
import inspect

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        duration_ms = round((te - ts) * 1000,2)
        signature = inspect.signature(method)
        if "self" in signature.parameters:
            self = args[0]
            if hasattr(self,"stats") and isinstance(self.stats, dict):
                if method.__name__ not in self.stats:
                    self.stats[method.__name__] = [duration_ms]
                else:
                    self.stats[method.__name__].append(duration_ms)
            print('%s => %r  %2.2f ms' % (self.__class__.__name__, method.__name__, duration_ms))
        return result
    return timed
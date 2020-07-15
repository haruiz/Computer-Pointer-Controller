
def perf_counts(method):
    def perf_countsd(*args, **kw):
        result = method(*args, **kw)
        self = args[0]
        if hasattr(self,"_exec_net"):
            metrics = self._exec_net.requests[0].get_perf_counts()
            self.perf_counts = metrics
        return result
    return perf_countsd
class PipelineOpt(object):
    def __init__(self):
        self.source = None

    def __iter__(self):
        return self.generator()

    #generate data
    def generator(self):
        try:
            while self.has_next():
                data = next(self.source) if self.source else {}
                if self.filter(data):
                    yield self.map(data)
        except StopIteration:
            return

    # By overriding an__or__ operator, it is possible to create Unix like pipeline:
    def __or__(self, other):
        other.source = self.generator()
        return other

    # The filer function allows us to filter the data passing our pipelin
    def filter(self, data):
        return True

    # The map function gives us the possibility to manipulate (map)
    # the pipeline data or update the state of the step as in First class.
    def map(self, data):
        return data

    def has_next(self):
        return True

    def cleanup(self):
        pass
import poptorch

# TODO -- pop options
opts = poptorch.Options()
opts.deviceIterations(4)
# opts.setAvailableMemoryProportion({"IPU0": 0.5, "IPU0": 0.6, "IPU0": 0.6, "IPU0": 0.6})

def get_options():
    return opts

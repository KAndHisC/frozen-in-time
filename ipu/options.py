import poptorch


opts = poptorch.Options()
opts.deviceIterations(8)
opts.Training.gradientAccumulation(19)
# opts.setAvailableMemoryProportion({"IPU0": 0.5, "IPU0": 0.6, "IPU0": 0.6, "IPU0": 0.6})
opts.Precision.enableFloatingPointExceptions(True)
opts.outputMode(poptorch.OutputMode.All)

inf_opts = poptorch.Options()
inf_opts.deviceIterations(4)
inf_opts.Precision.enableFloatingPointExceptions(True)
def get_train_opts():
    return opts

def get_inf_opts():
    return inf_opts

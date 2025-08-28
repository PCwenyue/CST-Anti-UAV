from experiments import ExperimentCSTAntiUAV

#from utils.trackers import UAV_Trackers as Trackers
from utils.trackers import CST_Trackers as Trackers

evaluation_metrics=['State accuracy', 'Success plots', 'Precision plots']

dataset_path=''

# test or val
subset='test'

# Setting experimental parameters
experiment = ExperimentCSTAntiUAV(root_dir=dataset_path, subset=subset)

experiment.report(Trackers)

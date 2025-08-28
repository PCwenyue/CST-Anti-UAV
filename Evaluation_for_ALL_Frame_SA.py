from experiments import ExperimentCSTAntiUAVF

#from utils.trackers import UAV_Trackers as Trackers
from utils.trackers import CST_Trackers as Trackers

evaluation_metrics=['State accuracy']

dataset_path=''

# test or val
subset='test'

# Setting experimental parameters
experiment = ExperimentCSTAntiUAVF(root_dir=dataset_path, subset=subset)

experiment.report(Trackers)

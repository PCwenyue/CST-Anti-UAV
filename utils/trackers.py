import numpy as np
"""

mode 1 means the results is formatted by (x,y,w,h)
mode 2 means the results is formatted by (x1,y1,x2,y2)

UAV_Trackers indicates results trained with the AntiUAV410 dataset
CST_Trackers indicates results trained with the CSTAntiUAV dataset

"""
UAV_Trackers=[
{'name': 'AiATrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/aiatrack/baseline_410_CST/result', 'mode': 1},
{'name': 'AQATrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/aqatrack/AQATrack-ep150-full-256_410_CST/cstantiuav_test/aqatrack_hivitb_v1-150/result', 'mode': 1},
{'name': 'ARTrack_256', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/artrack_256/UAV_artrack_256_CST/result', 'mode': 1},
{'name': 'ARTrack_384', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/artrack_384/UAV_artrack_384_CST/result', 'mode': 1},
{'name': 'ARTrack2', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/artrack2/410_artrackv2_256_full_CST/result', 'mode': 1},
{'name': 'DropTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/droptrack/vitb_384_mae_ce_32x4_ep300_410_CST/result', 'mode': 1},
{'name': 'GlobalTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/globaltrack/results_410pth_CST/results', 'mode': 1},
{'name': 'GRM', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/grm/vitb_256_ep300_410_CST/result', 'mode': 1},
{'name': 'HIPTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/hiptrack/hiptrack_410_CST/result', 'mode': 1},
{'name': 'KYS', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/kys/default_410_CST/result', 'mode': 1},
{'name': 'Mixformer', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/mixformer/baseline_410_CST/result', 'mode': 1},
{'name': 'MixformerV2', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/mixformer2/student_288_depth8_410_CST/result', 'mode': 1},
{'name': 'OSTrack_256', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/ostrack_256/vitb_256_mae_32x4_ep300_410_CST/result', 'mode': 1},
{'name': 'OSTrack_384', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/ostrack_384/UAV_384_mae_ce_32x4_ep300/result', 'mode': 1},
{'name': 'PrDiMP', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/prdimp/prdimp50_410_CST/result', 'mode': 1},
{'name': 'Refocus-TIR', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/refocus/vitb_refocus_410_cstantiuav_ep60/result', 'mode': 1},
{'name': 'RomRrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/romtrack/baseline_stage2_410_CST/result', 'mode': 1},
{'name': 'SeqTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/seqtrack/410_seqtrack_b256/result', 'mode': 1},
{'name': 'SiamDT', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/siamdt/results_410_CST/results', 'mode': 1},
{'name': 'Stark_s', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/stark_s/baseline_410_CST/result', 'mode': 1},
{'name': 'Stark_st', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/stark_st/baseline_410_CST/result', 'mode': 1},
{'name': 'Tomp', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/Trained410/tomp/tomp50_410_CST/result', 'mode': 1},
 ]


CST_Trackers=[
{'name': 'STFTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/ststrack/qftrack_swin_tiny_sgd/results', 'mode': 1},
{'name': 'AiATrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/aiatrack/baseline_CST_CST/result', 'mode': 1},
{'name': 'AQATrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/aqatrack/AQATrack-ep150-full-256_CST_CST/cstantiuav_test/aqatrack_hivitb_v1-150/result', 'mode': 1},
{'name': 'ARTrack_256', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/artrack_256/CST_artrack_256_CST/result', 'mode': 1},
{'name': 'ARTrack_384', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/artrack_384/CST_artrack_384_CST/result', 'mode': 1},
{'name': 'ARTrackV2', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/artrack2/CST_artrackv2_256_full_CST/result', 'mode': 1},
{'name': 'DropTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/droptrack/vitb_384_mae_ce_32x4_ep300_CST_CST/result', 'mode': 1},
{'name': 'GlobalTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/globaltrack/results_CSTpth_CST/results', 'mode': 1},
{'name': 'GRM', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/grm/vitb_256_ep300_CST_CST/result', 'mode': 1},
{'name': 'HIPTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/hiptrack/hiptrack_CST_CST/result', 'mode': 1},
{'name': 'KYS', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/kys/default_CST_CST/result', 'mode': 1},
{'name': 'Mixformer', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/mixformer/baseline_CST_CST/result', 'mode': 1},
{'name': 'MixformerV2', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/mixformer2/student_288_depth8_CST_CST/result', 'mode': 1},
{'name': 'OSTrack_256', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/ostrack_256/vitb_256_mae_32x4_ep300_CST_CST/result', 'mode': 1},
{'name': 'OSTrack_384', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/ostrack_384/CST_vitb_384_mae_ce_32x4_ep300_CST/result', 'mode': 1},
{'name': 'PrDiMP', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/prdimp/prdimp50_CST_CST/result', 'mode': 1},
{'name': 'Refocus-TIR', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/refocus/vitb_refocus_CST_cstantiuav_ep60/result', 'mode': 1},
{'name': 'RomTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/romtrack/baseline_stage2_CST_CST/result', 'mode': 1},
{'name': 'SeqTrack', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/seqtrack/CST_seqtrack_b256_CST/result', 'mode': 1},
{'name': 'SiamDT', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/siamdt/results_CST_CST/results', 'mode': 1},
{'name': 'Stark_s', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/stark_s/baseline_CST_CST/result', 'mode': 1},
{'name': 'Stark_st', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/stark_st/baseline_CST_CST/result', 'mode': 1},
{'name': 'Tomp', 'path': 'D:/A405/project/benchmark/Tracking_results_CST/TrainedCST/tomp/tomp50_CST_CST/result', 'mode': 1},
  ]
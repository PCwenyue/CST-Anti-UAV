from __future__ import absolute_import, division, print_function

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from PIL import Image

from datasets import CSTAntiUAV
from utils.metrics import rect_iou, center_error
from utils.viz import show_frame


class ExperimentCSTAntiUAVF(object):
    r"""Experiment pipeline and evaluation toolkit for CSTAntiUAV dataset.
    
    Args:
        root_dir (string): Root directory of CSTAntiUAV dataset.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, subset,
                 result_dir='results', report_dir='reports_frame', start_idx=0, end_idx=None):
        super(ExperimentCSTAntiUAVF, self).__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.dataset = CSTAntiUAV(os.path.join(root_dir, subset))
        self.result_dir = os.path.join(result_dir, 'CSTAntiUAV', subset)
        self.report_dir = os.path.join(report_dir, 'CSTAntiUAV', subset)
        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.use_confs = True
        self.dump_as_csv = False
        self.att_name = ['Occlusion','Out-of-View', 'Scale Variation','Thermal Crossover', 
                    'Dynamic Background Clutter','Complex Dynamic Background','Fast Motion',  
                    'Tiny Size', 'Small Size', 'Medium Size', 'Normal Size']
        self.att_fig_name = ['OC','OV', 'TC',  'SV','DBC', 'CDB','FM', 
                        'TS', 'SS', 'MS', 'NS']


    def iou(self, bbox1, bbox2): 
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
            bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, w1_1, h1_1) = bbox1
        (x0_2, y0_2, w1_2, h1_2) = bbox2
        x1_1 = x0_1 + w1_1
        x1_2 = x0_2 + w1_2
        y1_1 = y0_1 + h1_1
        y1_2 = y0_2 + h1_2
        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union

    def not_exist(self, pred):

        if len(pred) == 1 or len(pred) == 0:
            return 1.0
        else:
            return 0.0

    def eval(self, out_res, label_res):

        measure_per_frame = []

        for _pred, _gt, _exist in zip(out_res, label_res['gt'], label_res['exist']):

            if not _exist:
                measure_per_frame.append(self.not_exist(_pred))
            else:

                if len(_gt) < 4 or sum(_gt) == 0:
                    continue

                if len(_pred) == 4:
                    measure_per_frame.append(self.iou(_pred, _gt))
                else:
                    measure_per_frame.append(0.0)

                # try:
                #     measure_per_frame.append(iou(_pred, _gt))
                # except:
                #     measure_per_frame.append(0)

            # measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt))

        return measure_per_frame


    def report(self, trackers, plot_curves=False, plot_attcurves=False):

        assert isinstance(trackers, (list, tuple))

        if isinstance(trackers[0], dict):
            pass
        else:
            trackers = [
                {'name': trackers[0], 'path': os.path.join(
                self.result_dir, trackers[0]), 'mode': 1},
            ]

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir)
        if not os.path.isdir(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        tracker_names = []


        for tracker in trackers:
            name = tracker['name']
            mode = tracker['mode']
            tracker_names.append(name)
            print('Evaluating', name)

            performance.update({name: {
                'overall': {},
                'OC':{},
                'OV':{},
                'SV':{},
                'TC':{},
                'DBC':{},
                'CDB':{},
                'FM':{},
                'TS':{},
                'SS':{},
                'MS':{},
                'NS':{},
                'seq_wise': {},
                }})


            att_list = []

            for s, (_, label_res) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]
                record_file = os.path.join(
                    tracker['path'], '%s.txt' % seq_name)

                att_file = os.path.join(
                    'annos', self.subset, 'att', '%s.txt' % seq_name)
                with open(att_file, 'r') as f:
                    att_temp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
                att_list.append(att_temp)

                try:
                    with open(record_file, 'r') as f:
                        boxestemp = json.load(f)['res']
                except:
                    with open(record_file, 'r') as f:
                        boxestemp = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))

                if mode==2:
                    boxestemp[:, 2:] = boxestemp[:, 2:] - boxestemp[:, :2] + 1

                SA_Score = self.eval(boxestemp, label_res)

                frame_sa_file = os.path.join(report_dir, name, 'frame_state_accuracy_scores.txt')
                if not os.path.isdir(os.path.dirname(frame_sa_file)):
                    os.makedirs(os.path.dirname(frame_sa_file))
                with open(frame_sa_file, 'a', encoding='utf-8') as f:
                    for score in SA_Score:
                        f.write(str(score) + '\n')


                input = os.path.join(self.root_dir,self.subset)
                labels = self.read_labels(input)
             
                for key, values in labels.items():
                    label_file = os.path.join(report_dir, name, f'{key}.txt')
                  
                    if not os.path.isdir(os.path.dirname(label_file)):
                        os.makedirs(os.path.dirname(label_file))
                    with open(label_file, 'w', encoding='utf-8') as f:
                        for value in values:
                            f.write(f"{value}\n")
      
        average_scores = {}
        total_lines = 0

     
        for tracker in trackers:
            name = tracker['name']
         
            frame_sa_file = os.path.join(report_dir, name, 'frame_state_accuracy_scores.txt')
            with open(frame_sa_file, 'r', encoding='utf-8') as f:
                frame_scores = [float(line.strip()) for line in f.readlines()]

            total_lines = len(frame_scores)

          
            for key in labels.keys():
                label_file = os.path.join(report_dir, name, f'{key}.txt')
                
                with open(label_file, 'r', encoding='utf-8') as f:
                    values = [float(line.strip()) for line in f.readlines()]
                
                
                if total_lines > 0:
                    if key in ['OC']:
                        
                        valid_indices = [i for i in range(total_lines) if values[i] == 1]
                        if valid_indices:
                            for idx in valid_indices:
                                
                                relevant_indices = set()  
                                
                               
                                for i in range(max(0, idx - 1), idx):
                                    if values[i] == 0:  
                                        relevant_indices.add(i)
                                relevant_indices.add(idx)
                               
                                for i in range(idx + 1, min(total_lines, idx + 3)):
                                    if values[i] == 0:  
                                        relevant_indices.add(i)

                                if relevant_indices:
                                    average_scores[key] = sum(frame_scores[i] for i in relevant_indices) / len(relevant_indices)                                    
                                else:
                                    average_scores[key] = 0.0  

                    elif key in ['OV']:
                        valid_indices = [i for i in range(total_lines) if values[i] == 1]
                        if valid_indices:
                            for idx in valid_indices:
                              
                                relevant_indices = set()  
                             
                                for i in range(max(0, idx - 1), idx):
                                    if values[i] == 0: 
                                        relevant_indices.add(i)
                                relevant_indices.add(idx)
                                for i in range(idx + 1, min(total_lines, idx + 91)):
                                    if values[i] == 0: 
                                        relevant_indices.add(i)

                            
                                if relevant_indices:
                         
                                    average_scores[key] = sum(frame_scores[i] for i in relevant_indices) / len(relevant_indices)
                                else:
                                    average_scores[key] = 0.0  
            

                    else:
                      
                        count_valid_values = sum(1 for v in values if v == 1)  
                        if count_valid_values > 0:
                            if total_lines == len(values):
                                average_scores[key] = sum(frame_scores[i] * values[i] for i in range(total_lines)) / count_valid_values
 
            output_file = os.path.join(report_dir, name + '_frame_sa.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                for key, avg_score in average_scores.items():
                    f.write(f"{key}: {avg_score}\n")
            average_scores.clear()



    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))
            
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, '%s.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')
            
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times, confs=None):
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        if confs is not None:
            lines = ['%.4f' % c for c in confs]
            lines[0] = ''

            conf_file = record_file.replace(".txt", "_confidence.value")
            with open(conf_file, 'w') as f:
                f.write(str.join('\n', lines))

        # record running times
        time_dir = os.path.join(record_dir, 'times')
        if not os.path.isdir(time_dir):
            os.makedirs(time_dir)
        time_file = os.path.join(time_dir, os.path.basename(
            record_file).replace('.txt', '_time.txt'))
        np.savetxt(time_file, times, fmt='%.8f')

    def read_labels(self, input):
        labels = {
            'OC': [],
            'OV': [],
            'SV': [],
            'TC': [],
            'DBC': [],
            'CDB': [],
            'FM': [],
            'TS': [],
            'SS': [],
            'MS': [],
            'NS': []
        }
        
        
        for root, dirs, files in os.walk(input):
            for filename in files:
                if filename == "IR_label.json":
                    file_path = os.path.join(root, filename)
                    
                    if not os.path.exists(file_path):
                        print(f"not exist: {file_path}")
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"wrong JSON: {file_path}")
                            continue

                   
                    for key in labels.keys():
                        if key in data:
                            labels[key].extend(data[key])

        return labels

  